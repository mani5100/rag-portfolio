"use client";

import { Bot, Briefcase, Code2, Globe, Menu, Users, X } from "lucide-react";

interface NavbarProps {
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}

const SOCIAL_LINKS = [
  {
    label: "GitHub",
    href: "https://github.com/mani5100/",
    icon: Code2,
  },
  {
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/abdulrehman-shoukat",
    icon: Users,
  },
  {
    label: "Upwork",
    href: "https://upwork.com/freelancers/abdulrehmanshoukat",
    icon: Briefcase,
  },
  {
    label: "Website",
    href: "https://abdulrehmanshoukat.com/",
    icon: Globe,
  },
] as const;

export function Navbar({ sidebarOpen, onToggleSidebar }: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 flex h-14 items-center justify-between border-b border-[#2a2d3a] bg-[#0f1117]/95 px-4 backdrop-blur">
      {/* Hamburger (mobile only) */}
      <button
        className="flex h-10 w-10 items-center justify-center rounded-lg text-[#64748b] hover:bg-[#1a1d27] hover:text-[#f1f5f9] md:hidden"
        onClick={onToggleSidebar}
        aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Brand */}
      <div className="flex flex-1 min-w-0 items-center gap-2 md:flex-none">
        <Bot className="h-5 w-5 shrink-0 text-[#06b6d4]" />
        <span className="text-sm font-semibold tracking-wide text-[#06b6d4]">
          RAG Portfolio
        </span>
      </div>

      {/* Social links */}
      <nav className="flex items-center gap-1">
        {SOCIAL_LINKS.map(({ label, href, icon: Icon }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={label}
            className="group flex items-center gap-0 overflow-hidden rounded-full border border-[#2a2d3a] bg-[#1a1d27] px-2 py-2
                       transition-all duration-300 ease-in-out
                       hover:gap-2 hover:border-[#06b6d4]/40 hover:bg-[#06b6d4]/5 hover:px-3"
          >
            <Icon className="h-4 w-4 shrink-0 text-[#64748b] transition-colors duration-200 group-hover:text-[#06b6d4]" />
            <span
              className="max-w-0 overflow-hidden whitespace-nowrap text-xs font-medium text-[#f1f5f9] opacity-0
                         transition-all duration-300 ease-in-out
                         group-hover:max-w-[80px] group-hover:opacity-100"
            >
              {label}
            </span>
          </a>
        ))}
      </nav>
    </header>
  );
}
