import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { Banner } from "@/components/Banner";
import { Nav } from "@/components/Nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OvaScan v0.2",
  description:
    "Ovarian cyst segmentation in ultrasound: enhancement, restoration, " +
    "registration and visualization. " +
    "Research prototype, not a medical device.",
};

/** The banner and nav moved here from page.tsx so every route carries them.
 *  The disclaimer in particular must be on every screen, not only the one the
 *  user happened to land on first. */
export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#0b0f14] text-white">
        <Banner />
        <Nav />
        <div className="flex-1">{children}</div>

        <footer className="mx-auto w-full max-w-6xl px-4 py-6 text-xs leading-relaxed text-white/30">
          OvaScan v0.2 · YOLOv11n-seg · academic prototype for 21CSE428T Healthcare Analytics.
          Not a medical device. Not for clinical use. No image is stored.
        </footer>
      </body>
    </html>
  );
}
