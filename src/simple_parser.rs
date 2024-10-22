use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0},
    combinator::recognize,
    multi::{fold_many0, many0},
    number::complete::recognize_float,
    sequence::{delimited, pair},
    IResult,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(f64),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
}

fn term(input: &str) -> IResult<&str, Expression> {
    alt((number, ident, parens))(input)
}

fn ident(input: &str) -> IResult<&str, Expression> {
    let (r, res) = delimited(multispace0, identifier, multispace0)(input)?;
    Ok((r, Expression::Ident(res)))
}

fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn number(input: &str) -> IResult<&str, Expression> {
    let (r, v) = delimited(multispace0, recognize_float, multispace0)(input)?;
    Ok((
        r,
        Expression::NumLiteral(v.parse().map_err(|_| {
            nom::Err::Error(nom::error::Error {
                input,
                code: nom::error::ErrorKind::Digit,
            })
        })?),
    ))
}

fn parens(i: &str) -> IResult<&str, Expression> {
    delimited(
        multispace0,
        delimited(tag("("), expr, tag(")")),
        multispace0,
    )(i)
}

pub fn expr(i: &str) -> IResult<&str, Expression> {
    let (i, init) = term(i)?;

    fold_many0(
        pair(delimited(multispace0, char('+'), multispace0), term),
        move || init.clone(),
        |acc, (_op, val): (char, Expression)| Expression::Add(Box::new(acc), Box::new(val)),
    )(i)
}
