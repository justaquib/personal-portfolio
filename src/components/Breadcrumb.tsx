"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback } from "react";

export default function Breadcrumb() {
  const pathname = usePathname();
  const router = useRouter();

  const crumbs = pathname
    .split("/")
    .filter(Boolean)
    .map((segment, i, all) => {
      const href = "/" + all.slice(0, i + 1).join("/");
      return { label: decodeURIComponent(segment), href, id: segment };
    });

  const handleClick = useCallback(
    (e: React.MouseEvent, href: string, id: string) => {
      e.preventDefault();

      router.replace(href + "#" + id, { scroll: false });

      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    },
    [router]
  );

  return (
    <nav aria-label="Breadcrumb" className="text-sm">
      <ol className="flex space-x-0">
        <li>
          <Link href="/" onClick={(e) => handleClick(e, "/", "")}>
            Home
          </Link>
        </li>
        {crumbs.map((crumb) => (
          <li key={crumb.href} className="flex items-center">
            <span className="mx-1">/</span>
            <Link
              href={crumb.href + `#${crumb.id}`}
              onClick={(e) => handleClick(e, crumb.href, crumb.id)}
              className="hover:underline text-white underline"
            >
              {crumb.label}
            </Link>
          </li>
        ))}
      </ol>
    </nav>
  );
}
