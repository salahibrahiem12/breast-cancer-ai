import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-pink-200/70 bg-white/90">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-8 text-xs text-muted sm:flex-row sm:items-center sm:justify-between lg:px-8">
        <div className="space-y-1">
          <p className="font-semibold text-ink flex items-center gap-1.5">
            <span className="text-pink-primary font-bold">BreastCare AI</span> · Experimental Medical Decision-Support Research
          </p>
          <p className="text-slate-500 max-w-xl">
            This system is an investigational research prototype for breast ultrasound and mammography imaging workflows.
            It does not provide medical diagnoses, treatment prescriptions, or staging. All clinical decisions require qualified specialist physician evaluation.
          </p>
        </div>
        <div className="flex flex-wrap gap-4 text-xs font-medium">
          <Link href="/cancer-info" className="hover:text-pink-primary transition-colors">Cancer Information</Link>
          <Link href="/treatment-info" className="hover:text-pink-primary transition-colors">Clinical Guidelines</Link>
          <Link href="/model-evidence" className="hover:text-pink-primary transition-colors">Evidence & Datasets</Link>
          <Link href="/about" className="hover:text-pink-primary transition-colors">Safety Limitations</Link>
        </div>
      </div>
    </footer>
  );
}
