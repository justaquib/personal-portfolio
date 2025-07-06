// src/components/AnimatedList.tsx
"use client";

import React, {
  UIEvent,
  useRef,
  useState,
  useEffect,
} from "react";
import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AnimatedItem } from "./AnimatedItems";

export interface AnimatedListProps<T> {
  items: T[];
  getKey?: (item: T, index: number) => string;
  getHref?: (item: T, index: number) => string | undefined;
  onItemSelect?: (item: T, index: number) => void;
  renderItem?: (
    item: T,
    index: number,
    isSelected: boolean
  ) => React.ReactNode;
  showGradients?: boolean;
  enableArrowNavigation?: boolean;
  className?: string;
  itemClassName?: string;
  displayScrollbar?: boolean;
  initialSelectedIndex?: number;
}

export function AnimatedList<T>({
  items,
  getKey,
  getHref,
  onItemSelect,
  renderItem,
  showGradients = true,
  enableArrowNavigation = true,
  className = "",
  itemClassName = "",
  displayScrollbar = true,
  initialSelectedIndex = -1,
}: AnimatedListProps<T>) {
  const listRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const [selectedIndex, setSelectedIndex] = useState<number>(
    initialSelectedIndex
  );
  const [keyboardNav, setKeyboardNav] = useState(false);
  const [topOpacity, setTopOpacity] = useState(0);
  const [botOpacity, setBotOpacity] = useState(1);

  // scroll gradients
  const onScroll = (e: UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } =
      e.currentTarget;
    setTopOpacity(Math.min(scrollTop / 50, 1));
    const dist = scrollHeight - (scrollTop + clientHeight);
    setBotOpacity(
      scrollHeight <= clientHeight
        ? 0
        : Math.min(dist / 50, 1)
    );
  };

  useEffect(() => {
    if (!enableArrowNavigation) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setKeyboardNav(true);
        setSelectedIndex((i) =>
          Math.min(i + 1, items.length - 1)
        );
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setKeyboardNav(true);
        setSelectedIndex((i) => Math.max(i - 1, 0));
      }
      if (e.key === "Enter" && selectedIndex >= 0) {
        e.preventDefault();
        const item = items[selectedIndex];
        const href = getHref?.(item, selectedIndex);
        if (href) {
          router.push(href);
        } else {
          onItemSelect?.(item, selectedIndex);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
    };
  }, [
    enableArrowNavigation,
    items,
    selectedIndex,
    getHref,
    onItemSelect,
    router,
  ]);

  // scroll into view on keyboard nav
  useEffect(() => {
    if (!keyboardNav || selectedIndex < 0) return;
    const container = listRef.current;
    if (!container) return;

    const sel = container.querySelector(
      `[data-index="${selectedIndex}"]`
    ) as HTMLElement | null;
    if (sel) {
      const margin = 50;
      const top = sel.offsetTop,
        bottom = top + sel.offsetHeight;
      const viewTop = container.scrollTop + margin;
      const viewBot =
        container.scrollTop +
        container.clientHeight -
        margin;
      if (top < viewTop) {
        container.scrollTo({
          top: top - margin,
          behavior: "smooth",
        });
      } else if (bottom > viewBot) {
        container.scrollTo({
          top:
            bottom -
            container.clientHeight +
            margin,
          behavior: "smooth",
        });
      }
    }
    setKeyboardNav(false);
  }, [selectedIndex, keyboardNav]);

  return (
    <div className={`relative w-full h-full ${className}`}>
      <div
        ref={listRef}
        className={clsx(
          "overflow-y-auto",
          "min-h-[200px] h-full p-2 md:p-4",
          displayScrollbar
            ? "[&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-700"
            : "scrollbar-hide"
        )}
        onScroll={onScroll}
        style={{
          scrollbarWidth: displayScrollbar
            ? "thin"
            : "none",
          scrollbarColor: "#444 transparent",
        }}
      >
        {items.map((item, idx) => {
          const isSelected = idx === selectedIndex;
          const key =
            getKey?.(item, idx) ?? String(idx);
          const href = getHref?.(item, idx);

          const content = renderItem
            ? renderItem(item, idx, isSelected)
            : (
              <div
                className={clsx(
                  "p-4 bg-gray-800 rounded",
                  isSelected && "bg-gray-700",
                  itemClassName
                )}
              >
                <span className="text-white">
                  {String(item)}
                </span>
              </div>
            );

          const handleClick = () => {
            setSelectedIndex(idx);
            if (href) {
              router.push(href);
            } else {
              onItemSelect?.(item, idx);
            }
          };

          return (
            <AnimatedItem
              key={key}
              index={idx}
              delay={0.05 * idx}
              isSelected={isSelected}
              onMouseEnter={() =>
                setSelectedIndex(idx)
              }
              onClick={handleClick}
            >
              {href ? <Link href={href}>{content}</Link> : content}
            </AnimatedItem>
          );
        })}
      </div>

      {showGradients && (
        <>
          <div
            className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-black to-transparent pointer-events-none"
            style={{ opacity: topOpacity }}
          />
          <div
            className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-black to-transparent pointer-events-none"
            style={{ opacity: botOpacity }}
          />
        </>
      )}
    </div>
  );
}
