/**
 * Truncates text to a specified maximum length, adding '...' if truncated
 * @param text - The text to truncate
 * @param maxLength - Maximum length including the '...' (default: 20)
 * @returns The truncated text
 */
export function truncateText(text: string, maxLength: number = 20): string {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - 3) + '...'
}