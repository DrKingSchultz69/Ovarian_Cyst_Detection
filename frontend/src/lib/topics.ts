/** The six syllabus topics this project covers, and where each one lives.
 *
 *  21CSE428T Healthcare Analytics. Unit-1 is the EHR half (T1-T3), Unit-2 the
 *  biomedical imaging half (T4-T6).
 *
 *  NOTE ON "HER": the official syllabus writes "Preprocessing of HER" in both
 *  the unit description and the tutorial list. It means EHR -- Electronic
 *  Health Records. The spelling is kept here only where the syllabus text is
 *  quoted verbatim; everywhere else this project writes EHR.
 */

export interface Topic {
  id: "T1" | "T2" | "T3" | "T4" | "T5" | "T6";
  title: string;
  syllabus: string;
  href: string;
  unit: 1 | 2;
}

export const TOPICS: Topic[] = [
  {
    id: "T1",
    title: "Understanding EHR",
    syllabus: "Understanding HER",
    href: "/ehr#t1",
    unit: 1,
  },
  {
    id: "T2",
    title: "EHR preprocessing — standardization, cleaning",
    syllabus: "Preprocessing of HER – Standardization, Data Cleaning",
    href: "/ehr#t2",
    unit: 1,
  },
  {
    id: "T3",
    title: "EHR preprocessing — redundancy, missing data",
    syllabus: "Preprocessing of HER – Redundant data removal, Missing data",
    href: "/ehr#t3",
    unit: 1,
  },
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
  { href: "/ehr", label: "EHR pipeline", topics: ["T1", "T2", "T3"] },
] as const;
