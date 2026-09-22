"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV } from "@/lib/topics";

/** Site navigation, with the syllabus topics each page covers.
 *
 *  The topic chips are not decoration: this project exists to cover T1-T6, and
 *  putting the mapping in the chrome means a reader never has to guess which
 *  page answers which topic.
 */
export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-white/10 bg-white/[0.02]">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-1 gap-y-1 px-4 py-2">
        {NAV.map((item) => {
          const active =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`group flex items-baseline gap-2 rounded-lg px-3 py-1.5 text-sm transition ${
                active
                  ? "bg-sky-400/10 text-sky-200"
                  : "text-white/60 hover:bg-white/[0.04] hover:text-white/85"
              }`}
            >
              {item.label}
              <span className="flex gap-1">
                {item.topics.map((t) => (
                  <span
                    key={t}
                    className={`rounded px-1 py-0.5 text-[10px] font-medium tabular-nums ${
                      active
                        ? "bg-sky-400/20 text-sky-200/90"
                        : "bg-white/[0.06] text-white/35 group-hover:text-white/50"
                    }`}
                  >
                    {t}
                  </span>
                ))}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
