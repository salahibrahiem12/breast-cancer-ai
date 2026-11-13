import type { Metadata } from "next";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "BreastCare AI | Medical Imaging Research Platform",
    template: "%s | BreastCare AI",
  },
  description:
    "An experimental breast ultrasound and mammography decision-support research platform with multi-stage validation and Grad-CAM explainability.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-pink-bg text-slate-text antialiased">
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
