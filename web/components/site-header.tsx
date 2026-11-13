import Link from "next/link";

const links = [
  { href: "/", label: "Home" },
  { href: "/analyze", label: "AI Analysis" },
  { href: "/cancer-info", label: "Breast Cancer Info" },
  { href: "/treatment-info", label: "Treatment Guide" },
  { href: "/model-evidence", label: "Model & Evidence" },
  { href: "/about", label: "About & Safety" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b border-pink-200/80 bg-white/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3.5 lg:px-8">
        <Link href="/" className="flex items-center gap-3 group" aria-label="BreastCare AI home">
          <span className="grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-pink-primary to-pink-accent text-white font-bold text-lg shadow-md group-hover:scale-105 transition-transform">
            🎗️
          </span>
          <span>
            <span className="block text-base font-bold tracking-tight text-ink flex items-center gap-1.5">
              BreastCare <span className="text-pink-primary">AI</span>
            </span>
            <span className="block text-xs font-medium text-pink-accent">Medical Imaging Research</span>
          </span>
        </Link>
        <nav aria-label="Main navigation" className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm font-medium text-slate-text">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="transition-colors hover:text-pink-primary py-1"
            >
              {link.label}
            </Link>
          ))}
          <Link
            href="/analyze"
            className="ml-2 hidden sm:inline-flex items-center rounded-xl bg-pink-primary px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-pink-dark transition-all"
          >
            Launch Analyzer
          </Link>
        </nav>
      </div>
    </header>
  );
}
