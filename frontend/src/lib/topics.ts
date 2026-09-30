/** The syllabus topics this project covers, and where each one lives.
 *
 *  21CSE428T Healthcare Analytics, Unit-2 -- the biomedical imaging half.
 *
 *  T1-T3 (Unit-1) are handled as a DATASET exercise rather than an
 *  application feature: selecting a valid dataset, standardising it, and
 *  removing redundancy. That work lives in data_prep/mmotu_audit.py and
 *  data_prep/mmotu_image_audit.py, which run against the MMOTU archive and
 *  report their findings to stdout. There is nothing for the web app to
 *  render, so there is no page for them.
 */

export interface Topic {
  id: "T4" | "T5" | "T6";
  title: string;
  syllabus: string;
  href: string;
  unit: 2;
}

export const TOPICS: Topic[] = [
  {
    id: "T4",
    title: "Enhancement, restoration, segmentation, compression",
    syllabus:
      "Biomedical Image Processing – enhancement, restoration, segmentation, Compression",
    href: "/imaging",
    unit: 2,
  },
  {
    id: "T5",
    title: "SIFT, RANSAC, CNN",
    syllabus: "Biomedical Image Analysis Techniques – SIFT, RANSAC, CNN",
    href: "/registration",
    unit: 2,
  },
  {
    id: "T6",
    title: "Visualization",
    syllabus: "Biomedical Image - Visualization",
    href: "/",
    unit: 2,
  },
];

export const NAV = [
  { href: "/", label: "Segmentation", topics: ["T4", "T5", "T6"] },
  { href: "/imaging", label: "Enhance & compress", topics: ["T4", "T6"] },
  { href: "/registration", label: "Registration", topics: ["T5", "T6"] },
] as const;
