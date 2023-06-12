use polars::{prelude::*};
use std::path::{PathBuf, Path};

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
    let (m,n) = df.shape();
    let text1 = Series::new("input", &["TEXT1: "]).extend_constant(AnyValue::Utf8("TEXT1: "), m-1)?;
    println!("text1 length:{}", text1.len());
    let target = df.column("target").unwrap().clone();
    println!("target length:{}", target.len());
    let chunk = polars::functions::concat_str([text1, target].as_ref(), " ")?;
    let df = df.with_column(chunk.clone())?;
    
    println!("{:?}", chunk);
    //println!("{df}");

    println!("{:?}", df["input"]);

    Ok(())
}

