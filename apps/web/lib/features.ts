// The eight analyses by their canonical names (docs/ARCHITECTURE.md §66), keyed by their Analyze tab. Every label in the
// UI uses these names; never a letter or a shorthand.
export const FEATURE = {
  papers: "Paper-to-Paper Diff",
  discrepancy: "Result Discrepancy Investigator",
  recipe: "Reproducibility Recipe Builder",
  plan: "Next Experiment Planner",
  claim: "Claim-to-Evidence Auditor",
  transfer: "Generalization and Transferability Analyzer",
  review: "Literature Review Completeness Auditor",
  benchmark: "Benchmark Reality Explorer",
} as const;

export type FeatureTab = keyof typeof FEATURE;
