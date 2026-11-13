import type { ReactNode } from "react";

export function ContentPage({
  eyebrow,
  title,
  intro,
  children,
}: {
  eyebrow: string;
  title: string;
  intro: string;
  children: ReactNode;
}) {
  return (
    <main className="page-shell mx-auto max-w-5xl px-5 py-12 sm:py-16 lg:px-8">
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-pink-deep">{eyebrow}</p>
      <h1 className="mt-4 max-w-4xl text-4xl font-semibold tracking-tight text-ink sm:text-5xl">{title}</h1>
      <p className="mt-5 max-w-3xl text-lg leading-8 text-muted">{intro}</p>
      <div className="mt-10 space-y-6">{children}</div>
    </main>
  );
}

export function ContentCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-3xl border border-line bg-white p-6 shadow-card sm:p-8">
      <h2 className="text-xl font-semibold text-ink">{title}</h2>
      <div className="mt-3 space-y-3 leading-7 text-muted">{children}</div>
    </section>
  );
}
