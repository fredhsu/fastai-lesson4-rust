use polars::{prelude::*, lazy::dsl::concat_list};
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
    println!("{:?}", df);
    let input_col = df.select(
        concat_str([df["context"].str_value(), df["target"].str_value()])
        );



    df.with_columns(Series::new("input","TEXT1: " + df["context"].clone()))?;

    println!("{:?}", df["input"]);

    Ok(())
}

