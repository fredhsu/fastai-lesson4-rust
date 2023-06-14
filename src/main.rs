use polars::prelude::*;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the CSV file into a Polars DataFrame
    let path = "./data";
    let filename = "train.csv";
    let mut path_buf = PathBuf::from(path);

    path_buf.push(filename);
    let path = path_buf.as_path();

    let mut df = CsvReader::from_path(path)?
        .has_header(true)
        .infer_schema(None)
        .finish()?;

    // Display the DataFrame
    //println!("{:?}", df);
    println!("shape: {:?}", df.shape());
    let (m, _n) = df.shape();
    let text1 =
        Series::new("input", &["TEXT1: "]).extend_constant(AnyValue::Utf8("TEXT1: "), m - 1)?;
    let context = df.column("context").unwrap().clone();

    let text2 =
        Series::new("input", &["TEXT2: "]).extend_constant(AnyValue::Utf8("TEXT2: "), m - 1)?;
    let target = df.column("target").unwrap().clone();

    let anch =
        Series::new("input", &["ANCH: "]).extend_constant(AnyValue::Utf8("ANCH: "), m - 1)?;
    let anchor = df.column("anchor").unwrap().clone();

    let chunk =
        polars::functions::concat_str([text1, context, text2, target, anch, anchor].as_ref(), " ")?;
    let df = df.with_column(chunk.clone())?;

    println!("{:?}", df["input"].get(0)?);

    // Getting stuck with tokenizer. Can I use an existing tokenizer that is supported by
    // a library? How do you build a tokenizer based on an existing model like the transformers
    // library in python/hugging face does?

    Ok(())
}
