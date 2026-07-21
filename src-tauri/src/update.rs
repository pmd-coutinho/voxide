use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

const RELEASES_URL: &str = "https://api.github.com/repos/pmd-coutinho/voxide/releases";
const RELEASE_PAGE_PREFIXES: [&str; 1] = ["https://github.com/pmd-coutinho/voxide/releases/"];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateCheckResult {
    pub has_update: bool,
    pub latest_version: Option<String>,
    pub release_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReleaseNote {
    pub version: String,
    pub title: String,
    pub notes: String,
    pub published_at: Option<String>,
    pub release_url: Option<String>,
    pub is_prerelease: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    draft: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ComparableVersion {
    numbers: [u64; 3],
    prerelease: Vec<PrereleaseIdentifier>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum PrereleaseIdentifier {
    Numeric(u64),
    Text(String),
}

impl Ord for ComparableVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numbers
            .cmp(&other.numbers)
            .then_with(|| compare_prerelease_identifiers(&self.prerelease, &other.prerelease))
    }
}

fn compare_prerelease_identifiers(
    left: &[PrereleaseIdentifier],
    right: &[PrereleaseIdentifier],
) -> Ordering {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => Ordering::Equal,
        // A stable release has higher precedence than an equivalent prerelease.
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            for (left_identifier, right_identifier) in left.iter().zip(right) {
                let ordering = match (left_identifier, right_identifier) {
                    (PrereleaseIdentifier::Numeric(left), PrereleaseIdentifier::Numeric(right)) => {
                        left.cmp(right)
                    }
                    (PrereleaseIdentifier::Numeric(_), PrereleaseIdentifier::Text(_)) => {
                        Ordering::Less
                    }
                    (PrereleaseIdentifier::Text(_), PrereleaseIdentifier::Numeric(_)) => {
                        Ordering::Greater
                    }
                    (PrereleaseIdentifier::Text(left), PrereleaseIdentifier::Text(right)) => {
                        left.cmp(right)
                    }
                };
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.len().cmp(&right.len())
        }
    }
}

impl PartialOrd for ComparableVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn parse_version(value: &str) -> Option<ComparableVersion> {
    let mut normalized = value.trim().trim_start_matches(['v', 'V']);
    if let Some((without_metadata, _)) = normalized.split_once('+') {
        normalized = without_metadata;
    }
    let (numbers, prerelease) = normalized
        .split_once('-')
        .map_or((normalized, None), |(numbers, prerelease)| {
            (numbers, Some(prerelease))
        });
    let mut parsed = [0; 3];
    let parts = numbers.split('.').collect::<Vec<_>>();
    if parts.len() < 2 {
        return None;
    }
    for (index, part) in parts.into_iter().take(3).enumerate() {
        if part.is_empty() || !part.chars().all(|character| character.is_ascii_digit()) {
            return None;
        }
        parsed[index] = part.parse().ok()?;
    }
    Some(ComparableVersion {
        numbers: parsed,
        prerelease: prerelease
            .map(|value| {
                value
                    .split('.')
                    .map(|identifier| match identifier.parse::<u64>() {
                        Ok(number) => PrereleaseIdentifier::Numeric(number),
                        Err(_) => PrereleaseIdentifier::Text(identifier.to_ascii_lowercase()),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    })
}

fn is_prerelease_release(release: &GitHubRelease) -> bool {
    release.prerelease
        || parse_version(&release.tag_name).is_some_and(|version| !version.prerelease.is_empty())
}

pub fn is_release_url(value: &str) -> bool {
    RELEASE_PAGE_PREFIXES
        .iter()
        .any(|prefix| value.starts_with(prefix))
}

pub async fn check_for_update(
    current_version: &str,
    include_prerelease: bool,
) -> Result<UpdateCheckResult, String> {
    let current = parse_version(current_version).ok_or_else(|| {
        format!("The installed application version is invalid: {current_version}")
    })?;
    let releases = fetch_releases(current_version).await?;

    let latest = releases
        .into_iter()
        .filter(|release| !release.draft && (include_prerelease || !is_prerelease_release(release)))
        .filter_map(|release| {
            let version = parse_version(&release.tag_name)?;
            Some((version, release))
        })
        .max_by(
            |(left_version, left_release), (right_version, right_release)| {
                left_version.cmp(right_version).then_with(|| {
                    left_release
                        .published_at
                        .as_deref()
                        .unwrap_or_default()
                        .cmp(right_release.published_at.as_deref().unwrap_or_default())
                })
            },
        );

    let Some((version, release)) = latest else {
        return Ok(UpdateCheckResult {
            has_update: false,
            latest_version: None,
            release_url: None,
        });
    };
    let has_update = version > current;
    Ok(UpdateCheckResult {
        has_update,
        latest_version: has_update.then_some(release.tag_name),
        release_url: has_update
            .then_some(release.html_url)
            .filter(|url| is_release_url(url)),
    })
}

pub async fn recent_release_notes(include_prerelease: bool) -> Result<Vec<ReleaseNote>, String> {
    let releases = fetch_releases(env!("CARGO_PKG_VERSION")).await?;
    Ok(release_notes_from_releases(releases, include_prerelease))
}

fn release_notes_from_releases(
    mut releases: Vec<GitHubRelease>,
    include_prerelease: bool,
) -> Vec<ReleaseNote> {
    releases.retain(|release| {
        !release.draft && (include_prerelease || !is_prerelease_release(release))
    });
    releases.sort_by(|left, right| {
        let left_version = parse_version(&left.tag_name);
        let right_version = parse_version(&right.tag_name);
        right_version.cmp(&left_version).then_with(|| {
            right
                .published_at
                .as_deref()
                .unwrap_or_default()
                .cmp(left.published_at.as_deref().unwrap_or_default())
        })
    });
    releases
        .into_iter()
        .filter(|release| parse_version(&release.tag_name).is_some())
        .take(6)
        .map(|release| {
            let is_prerelease = is_prerelease_release(&release);
            ReleaseNote {
                title: release
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|title| !title.is_empty())
                    .unwrap_or(&release.tag_name)
                    .to_owned(),
                notes: release
                    .body
                    .as_deref()
                    .map(str::trim)
                    .filter(|notes| !notes.is_empty())
                    .unwrap_or("No release notes available.")
                    .to_owned(),
                published_at: release.published_at,
                release_url: is_release_url(&release.html_url).then_some(release.html_url),
                version: release.tag_name,
                is_prerelease,
            }
        })
        .collect()
}

async fn fetch_releases(current_version: &str) -> Result<Vec<GitHubRelease>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .user_agent(format!("Voxide/{current_version}"))
        .build()
        .map_err(|error| format!("Could not prepare the update check: {error}"))?;
    client
        .get(RELEASES_URL)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|error| format!("Could not contact GitHub Releases: {error}"))?
        .error_for_status()
        .map_err(|error| format!("GitHub Releases rejected the update check: {error}"))?
        .json::<Vec<GitHubRelease>>()
        .await
        .map_err(|error| format!("GitHub Releases returned an invalid response: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{is_release_url, parse_version, release_notes_from_releases, GitHubRelease};

    #[test]
    fn compares_v_prefixed_two_part_and_prerelease_versions() {
        assert!(parse_version("v1.2").unwrap() > parse_version("1.1.99").unwrap());
        assert!(parse_version("1.2.0").unwrap() > parse_version("1.2.0-beta.1").unwrap());
        assert!(parse_version("1.2.1-beta").unwrap() > parse_version("1.2.0").unwrap());
        assert!(parse_version("1.2.0-beta.10").unwrap() > parse_version("1.2.0-beta.2").unwrap());
        assert!(parse_version("1.2.0-1").unwrap() < parse_version("1.2.0-alpha").unwrap());
        assert_eq!(parse_version("v1.2+build.42"), parse_version("1.2.0"));
        assert!(parse_version("1").is_none());
    }

    #[test]
    fn rejects_invalid_versions_and_untrusted_release_pages() {
        assert!(parse_version("release-1.0").is_none());
        assert_eq!(parse_version("1.2.3.4"), parse_version("1.2.3"));
        assert!(is_release_url(
            "https://github.com/pmd-coutinho/voxide/releases/tag/v1.2.3"
        ));
        assert!(!is_release_url("https://example.test/releases/tag/v1.2.3"));
    }

    #[test]
    fn recent_notes_follow_semantic_order_and_release_safety_rules() {
        let notes = release_notes_from_releases(
            vec![
                GitHubRelease {
                    tag_name: "v1.2.0".into(),
                    html_url: "https://github.com/pmd-coutinho/voxide/releases/tag/v1.2.0".into(),
                    name: Some("Stable release".into()),
                    body: Some("Useful fixes".into()),
                    published_at: Some("2026-07-20T12:00:00Z".into()),
                    prerelease: false,
                    draft: false,
                },
                GitHubRelease {
                    tag_name: "v1.3.0-beta.1".into(),
                    html_url: "https://github.com/pmd-coutinho/voxide/releases/tag/v1.3.0-beta.1"
                        .into(),
                    name: None,
                    body: None,
                    published_at: None,
                    // GitHub's boolean is not always set on a tag with a
                    // prerelease suffix, so the tag itself must still filter it.
                    prerelease: false,
                    draft: false,
                },
                GitHubRelease {
                    tag_name: "v1.1.0".into(),
                    html_url: "https://example.test/releases/tag/v1.1.0".into(),
                    name: None,
                    body: None,
                    published_at: None,
                    prerelease: false,
                    draft: false,
                },
            ],
            false,
        );

        assert_eq!(
            notes
                .iter()
                .map(|note| note.version.as_str())
                .collect::<Vec<_>>(),
            ["v1.2.0", "v1.1.0"]
        );
        assert_eq!(notes[0].title, "Stable release");
        assert_eq!(notes[1].notes, "No release notes available.");
        assert!(notes[1].release_url.is_none());

        let with_beta = release_notes_from_releases(
            vec![GitHubRelease {
                tag_name: "v1.3.0-beta.1".into(),
                html_url: "https://github.com/pmd-coutinho/voxide/releases/tag/v1.3.0-beta.1"
                    .into(),
                name: None,
                body: None,
                published_at: None,
                prerelease: true,
                draft: false,
            }],
            true,
        );
        assert_eq!(with_beta.len(), 1);
        assert!(with_beta[0].is_prerelease);
    }
}
