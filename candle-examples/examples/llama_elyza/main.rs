// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;
use std::env;
use std::path::PathBuf;

// ここで読み込んでるのが7bのモデルなので、7bじゃないとダメ？
use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};
use regex::Regex;

const EOS_TOKEN: &str = "</s>";

const DEFAULT_PROMPT: &str = "[INST]<<SYS>>\nあなたは高名な研究者で、自分の専門分野について精通しています。特にあなたは、文から研究キーワードを過不足なく抽出することが得意です。例えば、「自然言語処理における逆接の談話関係についてのアノテーション」というという文からは、「自然言語処理」「談話関係」「アノテーション」といったキーワードを抽出することができます。\n<</SYS>>\n\n「高速原子間力顕微鏡を用いた、タンパク質の一分子観察による動態解」という文から研究キーワードを抽出して、json形式で出力してください。[/INST]";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The model size to use.
    #[arg(long, default_value = "v2")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}


fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    match env::current_dir() {
        Ok(path) => {
            println!("現在のディレクトリは: {}", path.display());
        },
        Err(e) => {
            println!("エラー: {}", e);
        }
    }

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, cache) = {
        let models_dir: &str = "./candle-examples/examples/llama_elyza/models";

        let tokenizer_filename = PathBuf::from(format!("{}/tokenizer.json", models_dir));
        println!("tokenizer_filename : {}", tokenizer_filename.display());
        let config_filename = PathBuf::from(format!("{}/config.json", models_dir));
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filename_1: PathBuf = PathBuf::from(format!("{}/model-00001-of-00003.safetensors", models_dir));
        let filename_2: PathBuf = PathBuf::from(format!("{}/model-00002-of-00003.safetensors", models_dir));
        let filename_3: PathBuf = PathBuf::from(format!("{}/model-00003-of-00003.safetensors", models_dir));

        let filenames: Vec<PathBuf> = vec![filename_1, filename_2, filename_3];

        println!("building the model");
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &cache, &config)?, tokenizer_filename, cache)
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;

    let mut undecoded_tokens = "".to_string();
    let mut undecoded_tokens_vector = Vec::new();
    for index in 0..args.sample_len {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        // println!("{next_token}");
        if let Some(texts) = tokenizer.id_to_token(next_token) {
            let mut texts = texts.replace('▁', " ").replace("<0x0A>", "\n");
            if texts.starts_with("<0") {
                let hex_string = texts[1..5].to_string();
                let hex_string_trimmed = hex_string.trim_start_matches("0x");
                let u8_value: u8 = u8::from_str_radix(hex_string_trimmed, 16)?;
                undecoded_tokens_vector.push(u8_value);
            }
            if undecoded_tokens_vector.len() >= 3 {
                // println!("start decoding : {:?}", undecoded_tokens_vector);
                let decoded_token = String::from_utf8(undecoded_tokens_vector.clone())?;
                // println!("デコード結果 : {}", decoded_token);
                texts.push_str(&decoded_token);
                undecoded_tokens_vector = vec![];
            }
            // デコード前のトークン列を削除
            let re = Regex::new(r"<0x..>").unwrap();
            texts = re.replace_all(&texts, "").to_string();
            print!("{texts}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
