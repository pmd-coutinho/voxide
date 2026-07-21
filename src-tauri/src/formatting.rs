//! Deterministic dictation cleanup performed before optional AI enhancement.
//!
//! These transforms intentionally require a spoken prefix ("literal" by
//! default) for punctuation. That keeps normal prose such as "I said comma"
//! intact while allowing deliberate character entry without an LLM.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PunctuationRule {
    pub aliases: Vec<String>,
    pub symbol: String,
}

#[derive(Debug, Clone)]
pub struct OutputFormatting<'a> {
    pub remove_filler_words: bool,
    pub filler_words: &'a [String],
    pub auto_convert_punctuation: bool,
    pub punctuation_prefix: &'a str,
    pub punctuation_rules: &'a [PunctuationRule],
    pub literal_dictation_formatting: bool,
    pub lowercase_first_letter: bool,
    pub remove_trailing_period: bool,
}

pub fn default_filler_words() -> Vec<String> {
    [
        "um", "uh", "er", "ah", "eh", "umm", "uhh", "err", "ahh", "ehh", "hmm", "hm", "mm", "mmm",
        "erm", "urm", "ugh",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub fn default_punctuation_rules() -> Vec<PunctuationRule> {
    let rule = |symbol: &str, aliases: &[&str]| PunctuationRule {
        symbol: symbol.into(),
        aliases: aliases.iter().map(|alias| (*alias).into()).collect(),
    };
    vec![
        rule(",", &["comma"]),
        rule(".", &["period", "full stop"]),
        rule(".", &["dot"]),
        rule("?", &["question mark", "questionmark"]),
        rule("!", &["exclamation mark", "exclamation point", "bang"]),
        rule(":", &["colon"]),
        rule(";", &["semicolon", "semi colon"]),
        rule("...", &["ellipsis", "dot dot dot", "three dots"]),
        rule("/", &["slash", "forward slash", "forwardslash"]),
        rule("\\", &["backslash", "back slash"]),
        rule("-", &["hyphen"]),
        rule("-", &["dash", "minus sign"]),
        rule("—", &["em dash", "long dash"]),
        rule("–", &["en dash"]),
        rule(
            "(",
            &[
                "open parenthesis",
                "open parentheses",
                "left parenthesis",
                "left parentheses",
                "open paren",
                "left paren",
            ],
        ),
        rule(
            ")",
            &[
                "close parenthesis",
                "close parentheses",
                "right parenthesis",
                "right parentheses",
                "close paren",
                "right paren",
            ],
        ),
        rule(
            "[",
            &[
                "open bracket",
                "left bracket",
                "open square bracket",
                "left square bracket",
            ],
        ),
        rule(
            "]",
            &[
                "close bracket",
                "right bracket",
                "close square bracket",
                "right square bracket",
            ],
        ),
        rule(
            "{",
            &[
                "open brace",
                "left brace",
                "open curly brace",
                "left curly brace",
                "open curly bracket",
                "left curly bracket",
            ],
        ),
        rule(
            "}",
            &[
                "close brace",
                "right brace",
                "close curly brace",
                "right curly brace",
                "close curly bracket",
                "right curly bracket",
            ],
        ),
        rule(
            "<",
            &["open angle bracket", "left angle bracket", "less than sign"],
        ),
        rule(
            ">",
            &[
                "close angle bracket",
                "right angle bracket",
                "greater than sign",
            ],
        ),
        rule("\"", &["quote", "quotes", "quotation mark", "double quote"]),
        rule(
            "\"",
            &[
                "open quote",
                "opening quote",
                "open double quote",
                "opening double quote",
            ],
        ),
        rule(
            "\"",
            &[
                "close quote",
                "closing quote",
                "close double quote",
                "closing double quote",
            ],
        ),
        rule("'", &["single quote"]),
        rule("'", &["apostrophe"]),
        rule("@", &["at the rate", "at sign", "commercial at"]),
        rule("&", &["ampersand", "and sign"]),
        rule("+", &["plus sign", "plus"]),
        rule("=", &["equals sign", "equal sign", "equal", "equals"]),
        rule("%", &["percent sign", "percentage sign", "percent"]),
        rule("$", &["dollar sign", "dollar"]),
        rule(
            "#",
            &["hash", "hash sign", "hashtag", "pound sign", "number sign"],
        ),
        rule("*", &["asterisk", "star symbol"]),
        rule("_", &["underscore"]),
        rule("|", &["pipe", "vertical bar"]),
        rule("~", &["tilde"]),
        rule("^", &["caret"]),
        rule("`", &["backtick", "back tick"]),
    ]
}

/// Matches Voxide's settings-store normalization for editable spoken
/// punctuation: aliases are case-insensitive phrases, duplicate aliases in a
/// rule are discarded, and a rule needs both an alias and a symbol.
pub fn normalize_punctuation_rules(rules: Vec<PunctuationRule>) -> Vec<PunctuationRule> {
    rules
        .into_iter()
        .filter_map(|rule| {
            let mut aliases = Vec::with_capacity(rule.aliases.len());
            for alias in rule.aliases {
                let alias = alias.trim().to_lowercase();
                if !alias.is_empty() && !aliases.contains(&alias) {
                    aliases.push(alias);
                }
            }
            let symbol = rule.symbol.trim().to_owned();
            (!aliases.is_empty() && !symbol.is_empty())
                .then_some(PunctuationRule { aliases, symbol })
        })
        .collect()
}

/// Early Tauri builds accidentally merged a few source-default rules whose
/// spacing depends on their alias. Upgrade precisely that known default shape
/// without rewriting a user's custom punctuation dictionary.
pub fn migrate_legacy_port_punctuation_rules(rules: Vec<PunctuationRule>) -> Vec<PunctuationRule> {
    let rules = normalize_punctuation_rules(rules);
    let has_rule = |symbol: &str, aliases: &[&str]| {
        rules.iter().any(|rule| {
            rule.symbol == symbol
                && rule.aliases.len() == aliases.len()
                && rule
                    .aliases
                    .iter()
                    .zip(aliases)
                    .all(|(actual, expected)| actual == expected)
        })
    };
    if rules.len() == default_punctuation_rules().len().saturating_sub(5)
        && has_rule(".", &["period", "full stop", "dot"])
        && has_rule("-", &["hyphen", "dash", "minus sign"])
        && has_rule(
            "\"",
            &[
                "quote",
                "quotes",
                "quotation mark",
                "double quote",
                "open quote",
                "opening quote",
                "open double quote",
                "opening double quote",
                "close quote",
                "closing quote",
                "close double quote",
                "closing double quote",
            ],
        )
        && has_rule("'", &["single quote", "apostrophe"])
    {
        default_punctuation_rules()
    } else {
        rules
    }
}

pub fn normalize_punctuation_prefix(value: &str) -> Option<String> {
    let prefix = value.trim().to_lowercase();
    (!prefix.is_empty()).then_some(prefix)
}

pub fn apply_before_ai(text: &str, settings: &OutputFormatting<'_>) -> String {
    let text = if settings.remove_filler_words {
        remove_filler_words(text, settings.filler_words)
    } else {
        text.to_owned()
    };
    if settings.auto_convert_punctuation {
        apply_spoken_punctuation(
            &text,
            settings.punctuation_prefix,
            settings.punctuation_rules,
        )
    } else {
        text
    }
}

pub fn apply_final_output(text: &str, settings: &OutputFormatting<'_>) -> String {
    apply_final_output_with_context(text, settings, None, None)
}

pub fn apply_final_output_with_context(
    text: &str,
    settings: &OutputFormatting<'_>,
    application: Option<&str>,
    window_title: Option<&str>,
) -> String {
    let mut output = if settings.literal_dictation_formatting {
        apply_literal_dictation_formatting(text, application, window_title)
    } else {
        text.to_owned()
    };
    if settings.remove_trailing_period && output.ends_with('.') {
        output.pop();
    }
    if settings.lowercase_first_letter {
        output = lowercase_first_letter(&output);
    }
    output
}

/// Chains consecutive dictations naturally. This mirrors Voxide's
/// continuous-dictation formatting while accepting the best available text
/// before the cursor from the host runtime.
pub fn apply_continuous_dictation_formatting(
    text: &str,
    preceding_text: &str,
    spacing_enabled: bool,
    smart_capitalization_enabled: bool,
) -> String {
    if text.is_empty() || (!spacing_enabled && !smart_capitalization_enabled) {
        return text.to_owned();
    }
    let mut result = text.to_owned();
    if smart_capitalization_enabled {
        let boundary = last_capitalization_boundary(preceding_text);
        result = replace_first_letter(&result, |character| {
            if boundary.is_none_or(is_sentence_ending_punctuation) {
                character.to_uppercase().collect()
            } else {
                character.to_lowercase().collect()
            }
        });
    }
    if spacing_enabled {
        if preceding_text
            .chars()
            .last()
            .is_some_and(|character| !character.is_whitespace())
            && !result.chars().next().is_some_and(char::is_whitespace)
        {
            result.insert(0, ' ');
        }
        if !result.chars().last().is_some_and(char::is_whitespace) {
            result.push(' ');
        }
    }
    result
}

/// Some chat applications trigger slash-command or mention autocomplete only
/// when the token is the terminal text. Continuous dictation adds a trailing
/// space for chaining, so remove just that space for the reference app set.
pub fn apply_terminal_literal_autocomplete_spacing(
    text: &str,
    literal_formatting_enabled: bool,
    application: Option<&str>,
    window_title: Option<&str>,
) -> String {
    if !literal_formatting_enabled
        || !text
            .chars()
            .last()
            .is_some_and(|character| matches!(character, ' ' | '\t'))
    {
        return text.to_owned();
    }
    let without_trailing = text.trim_end_matches([' ', '\t']);
    if without_trailing.is_empty() {
        return text.to_owned();
    }
    let context = [application, window_title]
        .into_iter()
        .flatten()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join(" ");
    let is_slash_command_app = ["codex", "chatgpt", "claude", "cursor", "windsurf"]
        .iter()
        .any(|name| context.contains(name));
    if is_slash_command_app
        && without_trailing.strip_prefix('/').is_some_and(|token| {
            valid_slash_command(&token) && !token.contains(char::is_whitespace)
        })
    {
        return without_trailing.to_owned();
    }
    let is_mention_app = ["slack", "discord", "teams"]
        .iter()
        .any(|name| context.contains(name));
    if is_mention_app && has_terminal_mention(without_trailing) {
        return without_trailing.to_owned();
    }
    text.to_owned()
}

fn has_terminal_mention(text: &str) -> bool {
    let Some((prefix, mention)) = text.rsplit_once('@') else {
        return false;
    };
    if prefix
        .chars()
        .last()
        .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return false;
    }
    let words = mention.split_whitespace().collect::<Vec<_>>();
    !words.is_empty()
        && words.len() <= 3
        && words.iter().enumerate().all(|(index, word)| {
            valid_mention_name(word)
                && (index == 0
                    || word
                        .chars()
                        .next()
                        .is_some_and(|character| character.is_uppercase()))
        })
}

fn last_capitalization_boundary(text: &str) -> Option<char> {
    for character in text.chars().rev() {
        if character == '\n' || character == '\r' {
            return None;
        }
        if character.is_whitespace()
            || matches!(
                character,
                '"' | '\'' | '”' | '’' | '»' | '›' | ')' | ']' | '}' | '」' | '』'
            )
        {
            continue;
        }
        return Some(character);
    }
    None
}

fn is_sentence_ending_punctuation(character: char) -> bool {
    matches!(character, '.' | '!' | '?')
}

fn replace_first_letter(text: &str, transform: impl Fn(char) -> String) -> String {
    let Some((index, character)) = text
        .char_indices()
        .find(|(_, character)| character.is_alphabetic())
    else {
        return text.to_owned();
    };
    let end = index + character.len_utf8();
    format!("{}{}{}", &text[..index], transform(character), &text[end..])
}

fn remove_filler_words(text: &str, filler_words: &[String]) -> String {
    let fillers = filler_words
        .iter()
        .map(|word| word.trim().to_lowercase())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if fillers.is_empty() {
        return text.to_owned();
    }
    // The reference deliberately splits only ordinary spaces here. Preserve
    // other text (including newlines) for the punctuation renderer that runs
    // immediately afterward.
    text.split(' ')
        .filter(|word| !word.is_empty())
        .filter(|word| {
            let normalized = word
                .trim_matches(|character: char| character.is_ascii_punctuation())
                .to_lowercase();
            !fillers.iter().any(|filler| filler == &normalized)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Clone, Copy)]
enum Spacing {
    RightAttached,
    LeftAttached,
    NoSpaceAround,
    SpaceAround,
    ToggleDoubleQuote,
    ToggleSingleQuote,
}

#[derive(Debug, Clone)]
enum PunctuationToken {
    Word {
        original: String,
        normalized: String,
    },
    Text(String),
}

impl PunctuationToken {
    fn word(&self) -> Option<&str> {
        match self {
            Self::Word { normalized, .. } => Some(normalized),
            Self::Text(_) => None,
        }
    }

    fn text(&self) -> &str {
        match self {
            Self::Word { original, .. } => original,
            Self::Text(text) => text,
        }
    }

    fn is_horizontal_whitespace(&self) -> bool {
        matches!(self, Self::Text(text) if !text.is_empty() && text.chars().all(is_horizontal_whitespace))
    }
}

#[derive(Debug, Clone)]
enum PunctuationOutput {
    Text(String),
    Punctuation { symbol: String, spacing: Spacing },
}

impl PunctuationOutput {
    fn is_horizontal_whitespace(&self) -> bool {
        matches!(self, Self::Text(text) if !text.is_empty() && text.chars().all(is_horizontal_whitespace))
    }
}

#[derive(Debug)]
struct PhraseRule {
    words: Vec<String>,
    symbol: String,
    spacing: Spacing,
}

fn apply_spoken_punctuation(text: &str, prefix: &str, rules: &[PunctuationRule]) -> String {
    let prefix_words = phrase_words(prefix);
    if prefix_words.is_empty() || text.is_empty() || rules.is_empty() {
        return text.to_owned();
    }
    let phrase_rules = phrase_rules(rules);
    if phrase_rules.is_empty() {
        return text.to_owned();
    }
    let tokens = tokenize_punctuation_text(text);
    let mut output = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if let Some(alias_index) = index_after_phrase(&tokens, index, &prefix_words) {
            if let Some((rule, end)) = matching_phrase_rule(&tokens, alias_index, &phrase_rules) {
                output.push(PunctuationOutput::Punctuation {
                    symbol: rule.symbol.clone(),
                    spacing: rule.spacing,
                });
                index = end;
                continue;
            }
        }
        output.push(PunctuationOutput::Text(tokens[index].text().to_owned()));
        index += 1;
    }
    render_punctuation_output(&remove_generated_comma_noise(output))
}

fn phrase_words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(str::to_lowercase)
        .filter(|word| !word.is_empty())
        .collect()
}

fn phrase_rules(rules: &[PunctuationRule]) -> Vec<PhraseRule> {
    rules
        .iter()
        .flat_map(|rule| {
            let spacing = spacing_for_rule(rule);
            rule.aliases.iter().filter_map(move |alias| {
                let words = phrase_words(alias);
                (!words.is_empty()).then(|| PhraseRule {
                    words,
                    symbol: rule.symbol.clone(),
                    spacing,
                })
            })
        })
        .collect()
}

fn tokenize_punctuation_text(text: &str) -> Vec<PunctuationToken> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut building_word = None;
    for character in text.chars() {
        let is_word = character.is_alphanumeric();
        match building_word {
            None => {
                current.push(character);
                building_word = Some(is_word);
            }
            Some(previous_kind) if previous_kind == is_word => current.push(character),
            Some(previous_kind) => {
                push_punctuation_token(&mut tokens, std::mem::take(&mut current), previous_kind);
                current.push(character);
                building_word = Some(is_word);
            }
        }
    }
    if let Some(is_word) = building_word {
        push_punctuation_token(&mut tokens, current, is_word);
    }
    tokens
}

fn push_punctuation_token(tokens: &mut Vec<PunctuationToken>, text: String, is_word: bool) {
    if text.is_empty() {
        return;
    }
    if is_word {
        tokens.push(PunctuationToken::Word {
            normalized: text.to_lowercase(),
            original: text,
        });
    } else {
        tokens.push(PunctuationToken::Text(text));
    }
}

fn index_after_phrase(
    tokens: &[PunctuationToken],
    start: usize,
    words: &[String],
) -> Option<usize> {
    let mut cursor = start;
    for (word_index, expected) in words.iter().enumerate() {
        if word_index > 0 {
            cursor = skip_horizontal_whitespace(tokens, cursor)?;
        }
        (tokens.get(cursor)?.word()? == expected).then_some(())?;
        cursor += 1;
    }
    skip_horizontal_whitespace(tokens, cursor)
}

fn matching_phrase_rule<'a>(
    tokens: &[PunctuationToken],
    start: usize,
    rules: &'a [PhraseRule],
) -> Option<(&'a PhraseRule, usize)> {
    let mut candidates = rules
        .iter()
        .filter(|rule| {
            tokens.get(start).and_then(PunctuationToken::word)
                == rule.words.first().map(String::as_str)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right.words.len().cmp(&left.words.len()).then_with(|| {
            right
                .words
                .join(" ")
                .chars()
                .count()
                .cmp(&left.words.join(" ").chars().count())
        })
    });
    for rule in candidates {
        let mut cursor = start;
        let mut matched = true;
        for (word_index, expected) in rule.words.iter().enumerate() {
            if word_index > 0 {
                let Some(next) = skip_horizontal_whitespace(tokens, cursor) else {
                    matched = false;
                    break;
                };
                cursor = next;
            }
            if tokens.get(cursor).and_then(PunctuationToken::word) != Some(expected) {
                matched = false;
                break;
            }
            cursor += 1;
        }
        if matched {
            return Some((rule, cursor));
        }
    }
    None
}

fn skip_horizontal_whitespace(tokens: &[PunctuationToken], mut index: usize) -> Option<usize> {
    if !tokens.get(index)?.is_horizontal_whitespace() {
        return None;
    }
    while tokens
        .get(index)
        .is_some_and(PunctuationToken::is_horizontal_whitespace)
    {
        index += 1;
    }
    (index < tokens.len()).then_some(index)
}

fn spacing_for_rule(rule: &PunctuationRule) -> Spacing {
    let aliases = &rule.aliases;
    match rule.symbol.as_str() {
        "." if aliases.iter().any(|alias| alias == "dot") => Spacing::NoSpaceAround,
        "," | "." | "?" | "!" | ":" | ";" | "..." | ")" | "]" | "}" | ">" | "%" => {
            Spacing::RightAttached
        }
        "(" | "[" | "{" | "<" | "$" => Spacing::LeftAttached,
        "+" | "=" | "&" | "—" | "–" => Spacing::SpaceAround,
        "-" if aliases
            .iter()
            .any(|alias| alias == "dash" || alias == "minus sign") =>
        {
            Spacing::SpaceAround
        }
        "-" | "/" | "\\" | "@" | "#" | "*" | "_" | "|" | "~" | "^" | "`" => Spacing::NoSpaceAround,
        "\"" if aliases
            .iter()
            .any(|alias| alias.starts_with("open ") || alias.starts_with("opening ")) =>
        {
            Spacing::LeftAttached
        }
        "\"" if aliases
            .iter()
            .any(|alias| alias.starts_with("close ") || alias.starts_with("closing ")) =>
        {
            Spacing::RightAttached
        }
        "\"" => Spacing::ToggleDoubleQuote,
        "'" if aliases.iter().any(|alias| alias == "apostrophe") => Spacing::NoSpaceAround,
        "'" => Spacing::ToggleSingleQuote,
        _ => Spacing::NoSpaceAround,
    }
}

fn remove_generated_comma_noise(parts: Vec<PunctuationOutput>) -> Vec<PunctuationOutput> {
    parts
        .iter()
        .enumerate()
        .filter_map(|(index, part)| {
            let is_comma =
                matches!(part, PunctuationOutput::Punctuation { symbol, .. } if symbol == ",");
            (!is_comma || !should_remove_generated_comma(index, &parts)).then(|| part.clone())
        })
        .collect()
}

fn should_remove_generated_comma(index: usize, parts: &[PunctuationOutput]) -> bool {
    let previous = significant_output_before(index, parts);
    let next = significant_output_after(index, parts);
    if let (
        Some(PunctuationOutput::Punctuation {
            symbol: previous, ..
        }),
        Some(PunctuationOutput::Punctuation { symbol: next, .. }),
    ) = (previous, next)
    {
        return previous
            .chars()
            .next()
            .is_some_and(is_comma_cleanup_punctuation)
            && next
                .chars()
                .next()
                .is_some_and(is_comma_cleanup_punctuation);
    }
    matches!(next, Some(PunctuationOutput::Punctuation { symbol, .. }) if symbol == "%")
        && matches!(previous, Some(PunctuationOutput::Text(text)) if text.chars().last().is_some_and(|character| character.is_ascii_digit()))
}

fn significant_output_before(
    index: usize,
    parts: &[PunctuationOutput],
) -> Option<&PunctuationOutput> {
    parts[..index]
        .iter()
        .rev()
        .find(|part| !part.is_horizontal_whitespace())
}

fn significant_output_after(
    index: usize,
    parts: &[PunctuationOutput],
) -> Option<&PunctuationOutput> {
    parts[index + 1..]
        .iter()
        .find(|part| !part.is_horizontal_whitespace())
}

fn is_comma_cleanup_punctuation(character: char) -> bool {
    matches!(
        character,
        '+' | '='
            | '%'
            | '-'
            | '—'
            | '–'
            | '/'
            | '\\'
            | '@'
            | '#'
            | '$'
            | '&'
            | '*'
            | '_'
            | '|'
            | '~'
            | '^'
            | '<'
            | '>'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '"'
            | '\''
            | '`'
            | '.'
            | '?'
            | '!'
            | ':'
            | ';'
    )
}

fn render_punctuation_output(parts: &[PunctuationOutput]) -> String {
    let mut output = String::new();
    let mut index = 0;
    let mut should_open_double_quote = true;
    let mut should_open_single_quote = true;
    while index < parts.len() {
        match &parts[index] {
            PunctuationOutput::Text(text) => {
                output.push_str(text);
                index += 1;
            }
            PunctuationOutput::Punctuation { symbol, spacing } => {
                let spacing = match spacing {
                    Spacing::ToggleDoubleQuote => {
                        let resolved = if should_open_double_quote {
                            Spacing::LeftAttached
                        } else {
                            Spacing::RightAttached
                        };
                        should_open_double_quote = !should_open_double_quote;
                        resolved
                    }
                    Spacing::ToggleSingleQuote => {
                        let resolved = if should_open_single_quote {
                            Spacing::LeftAttached
                        } else {
                            Spacing::RightAttached
                        };
                        should_open_single_quote = !should_open_single_quote;
                        resolved
                    }
                    spacing => *spacing,
                };
                match spacing {
                    Spacing::RightAttached => {
                        trim_trailing_horizontal_whitespace(&mut output);
                        output.push_str(symbol);
                        index += 1;
                    }
                    Spacing::LeftAttached => {
                        output.push_str(symbol);
                        index = skip_output_horizontal_whitespace(index, parts);
                    }
                    Spacing::NoSpaceAround => {
                        trim_trailing_horizontal_whitespace(&mut output);
                        output.push_str(symbol);
                        index = skip_output_horizontal_whitespace(index, parts);
                    }
                    Spacing::SpaceAround => {
                        trim_trailing_horizontal_whitespace(&mut output);
                        if !output.is_empty() && !output.ends_with('\n') && !output.ends_with('\r')
                        {
                            output.push(' ');
                        }
                        output.push_str(symbol);
                        index = skip_output_horizontal_whitespace(index, parts);
                        if has_following_non_whitespace_output(index, parts) {
                            output.push(' ');
                        }
                    }
                    Spacing::ToggleDoubleQuote | Spacing::ToggleSingleQuote => index += 1,
                }
            }
        }
    }
    output
}

fn trim_trailing_horizontal_whitespace(value: &mut String) {
    while value.chars().last().is_some_and(is_horizontal_whitespace) {
        value.pop();
    }
}

fn skip_output_horizontal_whitespace(mut index: usize, parts: &[PunctuationOutput]) -> usize {
    index += 1;
    while parts
        .get(index)
        .is_some_and(PunctuationOutput::is_horizontal_whitespace)
    {
        index += 1;
    }
    index
}

fn has_following_non_whitespace_output(index: usize, parts: &[PunctuationOutput]) -> bool {
    parts[index..].iter().any(|part| match part {
        PunctuationOutput::Text(text) => text
            .chars()
            .any(|character| !is_horizontal_whitespace(character)),
        PunctuationOutput::Punctuation { .. } => true,
    })
}

fn is_horizontal_whitespace(character: char) -> bool {
    character.is_whitespace()
        && !matches!(
            character,
            '\n' | '\r' | '\u{000B}' | '\u{000C}' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
}

fn apply_literal_dictation_formatting(
    text: &str,
    application: Option<&str>,
    window_title: Option<&str>,
) -> String {
    let words = text.split_whitespace().collect::<Vec<_>>();
    let mut output = Vec::with_capacity(words.len());
    let relaxed_mentions = is_relaxed_mention_app(application, window_title);
    let mut index = 0;
    while index < words.len() {
        let lower = words[index].to_ascii_lowercase();
        let (slash_start, token_start, needs_spoken_context) = match lower.as_str() {
            "slash" => (true, index + 1, true),
            "forward"
                if words
                    .get(index + 1)
                    .is_some_and(|word| word.eq_ignore_ascii_case("slash")) =>
            {
                (true, index + 2, true)
            }
            "/" => (true, index + 1, false),
            _ => (false, index, false),
        };
        if slash_start {
            if let Some(token) = words
                .get(token_start)
                .filter(|token| valid_slash_command(token))
                .filter(|_| !needs_spoken_context || has_spoken_slash_command_context(&output))
            {
                output.push(format!("/{}", token.to_ascii_lowercase()));
                index = token_start + 1;
                continue;
            }
        }

        let mention = match lower.as_str() {
            "tag" | "mention" => mention_name_at(&words, index + 1, false),
            "at" if words
                .get(index + 1)
                .is_some_and(|word| word.eq_ignore_ascii_case("sign")) =>
            {
                mention_name_at(&words, index + 2, false)
            }
            "at" if words
                .get(index + 1)
                .is_some_and(|word| word.eq_ignore_ascii_case("the"))
                && words
                    .get(index + 2)
                    .is_some_and(|word| word.eq_ignore_ascii_case("rate")) =>
            {
                mention_name_at(&words, index + 3, false)
            }
            "at" if relaxed_mentions && has_relaxed_mention_context(&output) => {
                mention_name_at(&words, index + 1, true)
            }
            _ => None,
        };
        if let Some((name, end)) = mention {
            output.push(format!("@{name}"));
            index = end;
            continue;
        }

        output.push(words[index].to_owned());
        index += 1;
    }
    output.join(" ")
}

fn mention_name_at(words: &[&str], start: usize, relaxed: bool) -> Option<(String, usize)> {
    let first = words.get(start)?;
    if !valid_mention_name(first) || (relaxed && !starts_with_ascii_uppercase(first)) {
        return None;
    }
    let mut tokens = vec![*first];
    let mut end = start + 1;
    while tokens.len() < 3 {
        let Some(next) = words.get(end) else {
            break;
        };
        if !valid_mention_name(next) || !starts_with_ascii_uppercase(next) {
            break;
        }
        tokens.push(next);
        end += 1;
    }
    Some((tokens.join(" "), end))
}

fn starts_with_ascii_uppercase(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_uppercase())
}

fn has_relaxed_mention_context(output: &[String]) -> bool {
    let Some(previous) = output.last() else {
        return true;
    };
    if previous
        .chars()
        .last()
        .is_some_and(|character| matches!(character, '.' | '!' | '?' | ':' | ';' | '(' | '[' | '{'))
    {
        return true;
    }
    matches!(
        previous
            .trim_matches(|character: char| !character.is_ascii_alphanumeric())
            .to_ascii_lowercase()
            .as_str(),
        "add"
            | "ask"
            | "cc"
            | "dm"
            | "hello"
            | "hey"
            | "hi"
            | "invite"
            | "message"
            | "notify"
            | "ping"
            | "send"
            | "tag"
            | "tell"
    )
}

fn has_spoken_slash_command_context(output: &[String]) -> bool {
    let Some(previous) = output.last() else {
        return true;
    };
    if previous
        .chars()
        .last()
        .is_some_and(|character| matches!(character, '.' | '!' | '?' | ':' | ';' | '(' | '[' | '{'))
    {
        return true;
    }
    matches!(
        previous
            .trim_matches(|character: char| !character.is_ascii_alphanumeric())
            .to_ascii_lowercase()
            .as_str(),
        "call"
            | "choose"
            | "do"
            | "enter"
            | "execute"
            | "open"
            | "pick"
            | "press"
            | "run"
            | "say"
            | "select"
            | "send"
            | "start"
            | "try"
            | "type"
            | "use"
            | "write"
    )
}

fn is_relaxed_mention_app(application: Option<&str>, window_title: Option<&str>) -> bool {
    [application, window_title]
        .into_iter()
        .flatten()
        .map(str::to_ascii_lowercase)
        .any(|context| {
            context.contains("slack") || context.contains("discord") || context.contains("teams")
        })
}

fn valid_slash_command(value: &&str) -> bool {
    let token = value.trim_matches(|character: char| character.is_ascii_punctuation());
    token.len() >= 2
        && token.len() <= 40
        && !matches!(
            token.to_ascii_lowercase().as_str(),
            "a" | "an"
                | "and"
                | "as"
                | "at"
                | "back"
                | "backslash"
                | "be"
                | "been"
                | "being"
                | "bin"
                | "by"
                | "comma"
                | "desktop"
                | "documents"
                | "dot"
                | "downloads"
                | "etc"
                | "for"
                | "forward"
                | "from"
                | "home"
                | "in"
                | "is"
                | "library"
                | "local"
                | "mark"
                | "of"
                | "on"
                | "or"
                | "period"
                | "private"
                | "question"
                | "quote"
                | "quotes"
                | "semicolon"
                | "slash"
                | "slashes"
                | "source"
                | "sources"
                | "src"
                | "the"
                | "tmp"
                | "to"
                | "user"
                | "users"
                | "usr"
                | "var"
                | "volumes"
                | "was"
                | "were"
                | "with"
                | "without"
        )
        && token.chars().enumerate().all(|(index, character)| {
            character.is_ascii_alphanumeric() || (index > 0 && matches!(character, '-' | '_'))
        })
        && token
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic())
}

fn valid_mention_name(value: &&str) -> bool {
    let token = value.trim_matches(|character: char| character.is_ascii_punctuation());
    !token.is_empty()
        && token.len() <= 64
        && !matches!(
            token.to_ascii_lowercase().as_str(),
            "a" | "an"
                | "airport"
                | "breakfast"
                | "brunch"
                | "class"
                | "dinner"
                | "home"
                | "hotel"
                | "house"
                | "lunch"
                | "meeting"
                | "night"
                | "noon"
                | "office"
                | "place"
                | "restaurant"
                | "school"
                | "shop"
                | "store"
                | "the"
                | "today"
                | "tomorrow"
                | "work"
                | "yesterday"
        )
        && token.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
        && token
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic())
}

fn lowercase_first_letter(text: &str) -> String {
    let Some(first) = text.chars().next() else {
        return String::new();
    };
    if !first.is_uppercase() {
        return text.to_owned();
    }
    let mut output = first.to_lowercase().collect::<String>();
    output.push_str(&text[first.len_utf8()..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> OutputFormatting<'static> {
        let filler_words = Box::leak(Box::new(default_filler_words()));
        let rules = Box::leak(Box::new(default_punctuation_rules()));
        OutputFormatting {
            remove_filler_words: true,
            filler_words,
            auto_convert_punctuation: true,
            punctuation_prefix: "literal",
            punctuation_rules: rules,
            literal_dictation_formatting: true,
            lowercase_first_letter: false,
            remove_trailing_period: false,
        }
    }

    #[test]
    fn removes_configured_fillers_before_other_processing() {
        let settings = settings();
        assert_eq!(
            apply_before_ai("Um hello er world", &settings),
            "hello world"
        );
    }

    #[test]
    fn converts_only_explicit_prefixed_punctuation() {
        let settings = settings();
        assert_eq!(
            apply_before_ai("hello literal comma world literal period", &settings),
            "hello, world."
        );
        assert_eq!(
            apply_before_ai("say comma out loud", &settings),
            "say comma out loud"
        );
    }

    #[test]
    fn default_punctuation_catalog_includes_every_reference_bracket_and_quote_alias() {
        let settings = settings();
        assert_eq!(
            apply_before_ai(
                "literal left parentheses x literal right parentheses literal left square bracket y literal right square bracket",
                &settings,
            ),
            "(x) [y]"
        );
        assert_eq!(
            apply_before_ai(
                "literal left curly bracket x literal right curly bracket literal opening double quote hello literal closing double quote",
                &settings,
            ),
            "{x} \"hello\""
        );
    }

    #[test]
    fn spoken_punctuation_uses_reference_rule_spacing_and_preserves_surrounding_text() {
        let settings = settings();
        assert_eq!(
            apply_before_ai(
                "name literal period next, literal open quote value literal close quote",
                &settings,
            ),
            "name. next, \"value\""
        );
        assert_eq!(
            apply_before_ai("api literal dot json", &settings),
            "api.json"
        );
        assert_eq!(
            apply_before_ai("foo literal hyphen bar literal dash baz", &settings),
            "foo-bar - baz"
        );

        let rules = vec![PunctuationRule {
            aliases: vec!["period".into(), "dot".into()],
            symbol: ".".into(),
        }];
        let mut customized = settings.clone();
        customized.punctuation_rules = &rules;
        assert_eq!(
            apply_before_ai("name literal period next", &customized),
            "name.next"
        );
        assert_eq!(
            apply_before_ai("50 literal comma literal percent", &settings),
            "50%"
        );
    }

    #[test]
    fn edited_punctuation_rules_match_the_reference_normalization_contract() {
        let rules = normalize_punctuation_rules(vec![
            PunctuationRule {
                aliases: vec!["  Comma  ".into(), "comma".into(), "".into()],
                symbol: " , ".into(),
            },
            PunctuationRule {
                aliases: vec!["ignored".into()],
                symbol: " ".into(),
            },
            PunctuationRule {
                aliases: vec![" ".into()],
                symbol: ".".into(),
            },
        ]);

        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].aliases, ["comma"]);
        assert_eq!(rules[0].symbol, ",");
        assert_eq!(
            normalize_punctuation_prefix("  LITERAL  ").as_deref(),
            Some("literal")
        );
        assert_eq!(normalize_punctuation_prefix(" \t"), None);
    }

    #[test]
    fn migrates_the_known_legacy_port_punctuation_default_without_touching_custom_rules() {
        let mut legacy = default_punctuation_rules();
        legacy.retain(|rule| !matches!(rule.symbol.as_str(), "." | "-" | "\"" | "'"));
        legacy.extend([
            PunctuationRule {
                aliases: vec!["period".into(), "full stop".into(), "dot".into()],
                symbol: ".".into(),
            },
            PunctuationRule {
                aliases: vec!["hyphen".into(), "dash".into(), "minus sign".into()],
                symbol: "-".into(),
            },
            PunctuationRule {
                aliases: vec![
                    "quote".into(),
                    "quotes".into(),
                    "quotation mark".into(),
                    "double quote".into(),
                    "open quote".into(),
                    "opening quote".into(),
                    "open double quote".into(),
                    "opening double quote".into(),
                    "close quote".into(),
                    "closing quote".into(),
                    "close double quote".into(),
                    "closing double quote".into(),
                ],
                symbol: "\"".into(),
            },
            PunctuationRule {
                aliases: vec!["single quote".into(), "apostrophe".into()],
                symbol: "'".into(),
            },
        ]);

        let migrated = migrate_legacy_port_punctuation_rules(legacy);
        assert_eq!(migrated.len(), default_punctuation_rules().len());
        assert!(migrated
            .iter()
            .any(|rule| rule.aliases == ["period", "full stop"]));
        assert!(migrated.iter().any(|rule| rule.aliases == ["dot"]));

        let custom = vec![PunctuationRule {
            aliases: vec!["my dot".into()],
            symbol: "•".into(),
        }];
        assert_eq!(migrate_legacy_port_punctuation_rules(custom)[0].symbol, "•");
    }

    #[test]
    fn supports_literal_slash_commands_and_mentions() {
        let settings = settings();
        assert_eq!(
            apply_final_output("slash status then tag Paul", &settings),
            "/status then @Paul"
        );
    }

    #[test]
    fn literal_command_and_mention_guards_avoid_reference_false_positives() {
        let settings = settings();
        assert_eq!(
            apply_final_output("please slash status", &settings),
            "please slash status"
        );
        assert_eq!(
            apply_final_output("run slash status", &settings),
            "run /status"
        );
        assert_eq!(apply_final_output("/ status", &settings), "/status");
        assert_eq!(
            apply_final_output("slash downloads then tag lunch", &settings),
            "slash downloads then tag lunch"
        );
    }

    #[test]
    fn supports_relaxed_mentions_only_in_reference_chat_contexts() {
        let settings = settings();
        assert_eq!(
            apply_final_output_with_context(
                "Hello at Ada Lovelace",
                &settings,
                Some("Slack"),
                None,
            ),
            "Hello @Ada Lovelace"
        );
        assert_eq!(
            apply_final_output_with_context("at Ada", &settings, Some("Notes"), None),
            "at Ada"
        );
        assert_eq!(
            apply_final_output_with_context("we met at Ada", &settings, Some("Discord"), None,),
            "we met at Ada"
        );
        assert_eq!(
            apply_final_output_with_context("Send at Ada", &settings, Some("Teams"), None,),
            "Send @Ada"
        );
    }

    #[test]
    fn applies_final_gaav_output_options() {
        let mut settings = settings();
        settings.lowercase_first_letter = true;
        settings.remove_trailing_period = true;
        assert_eq!(apply_final_output("Hello.", &settings), "hello");
    }

    #[test]
    fn chains_continuous_dictation_with_spacing_and_contextual_case() {
        assert_eq!(
            apply_continuous_dictation_formatting("hello", "A complete sentence. ", true, true),
            "Hello "
        );
        assert_eq!(
            apply_continuous_dictation_formatting("Hello", "A continuing clause", true, true),
            " hello "
        );
        assert_eq!(
            apply_continuous_dictation_formatting("hello", "", false, true),
            "Hello"
        );
    }

    #[test]
    fn preserves_terminal_chat_autocomplete_tokens() {
        assert_eq!(
            apply_terminal_literal_autocomplete_spacing("/status ", true, Some("Codex"), None,),
            "/status"
        );
        assert_eq!(
            apply_terminal_literal_autocomplete_spacing(
                "hello @Ada Lovelace ",
                true,
                Some("Slack"),
                None,
            ),
            "hello @Ada Lovelace"
        );
        assert_eq!(
            apply_terminal_literal_autocomplete_spacing("/status ", true, Some("Notes"), None,),
            "/status "
        );
    }
}
