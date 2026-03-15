'use client'

import { useEffect, useRef, forwardRef, useImperativeHandle, useState } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'

export interface QuillEditorHandle {
  getContent: () => string
  setContent: (content: string) => void
}

interface QuillEditorProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  height?: string
  onEnhance?: () => void
  isEnhancing?: boolean
}

const QuillEditor = forwardRef<QuillEditorHandle, QuillEditorProps>(({
  value,
  onChange,
  placeholder = 'Enter description...',
  height = '150px',
  onEnhance,
  isEnhancing
}, ref) => {
  const [isMounted, setIsMounted] = useState(false)
  const onChangeRef = useRef(onChange)
  const valueRef = useRef(value)
  
  // Keep refs updated
  useEffect(() => {
    onChangeRef.current = onChange
  }, [onChange])

  useEffect(() => {
    valueRef.current = value
  }, [value])

  // Initialize mounted state
  useEffect(() => {
    setIsMounted(true)
  }, [])

  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3],
        },
      }),
      Underline,
    ],
    content: value || '',
    onUpdate: ({ editor }) => {
      onChangeRef.current(editor.getHTML())
    },
    editorProps: {
      attributes: {
        class: 'prose prose-sm sm:prose lg:prose-lg xl:prose-2xl mx-auto focus:outline-none',
        'data-placeholder': placeholder,
        style: `min-height: ${height}`,
      },
    },
  })

  // Update content when value changes externally
  useEffect(() => {
    if (editor && value !== editor.getHTML()) {
      editor.commands.setContent(value || '')
    }
  }, [value, editor])

  useImperativeHandle(ref, () => ({
    getContent: () => {
      return editor?.getText()?.trim() || ''
    },
    setContent: (content: string) => {
      editor?.commands.setContent(content)
    }
  }), [editor])

  // Custom styling
  useEffect(() => {
    if (!isMounted) return
    
    const style = document.createElement('style')
    style.textContent = `
      .tiptap-editor {
        border: 1px solid #ced4da;
        border-radius: 0 0 0.5rem 0.5rem;
        background-color: #f8f9fa;
        padding: 0.75rem;
        min-height: ${height};
      }
      
      .tiptap-editor .ProseMirror {
        min-height: ${height};
        outline: none;
      }
      
      .tiptap-editor .ProseMirror p.is-editor-empty:first-child::before {
        color: #9ca3af;
        content: attr(data-placeholder);
        float: left;
        height: 0;
        pointer-events: none;
      }
      
      .tiptap-editor .ProseMirror h1,
      .tiptap-editor .ProseMirror h2,
      .tiptap-editor .ProseMirror h3 {
        color: #212529;
        margin-bottom: 0.5rem;
      }
      
      .tiptap-editor .ProseMirror p {
        margin-bottom: 0.5rem;
      }
      
      .tiptap-editor .ProseMirror ul,
      .tiptap-editor .ProseMirror ol {
        padding-left: 1.5rem;
        margin-bottom: 0.5rem;
      }
      
      .tiptap-toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        padding: 0.5rem;
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        border-radius: 0.5rem 0.5rem 0 0;
        margin-bottom: -1px;
      }
      
      .tiptap-toolbar button {
        padding: 0.25rem 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 0.25rem;
        background-color: white;
        color: #212529;
        cursor: pointer;
        font-size: 0.875rem;
        transition: all 0.2s;
      }
      
      .tiptap-toolbar button:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
      }
      
      .tiptap-toolbar button.is-active {
        background-color: #212529;
        color: white;
        border-color: #212529;
      }
      
      .tiptap-toolbar .separator {
        width: 1px;
        background-color: #ced4da;
        margin: 0 0.25rem;
      }
      
      .tiptap-toolbar button.enhancing {
        opacity: 0.7;
        cursor: not-allowed;
      }
    `
    document.head.appendChild(style)
    
    return () => {
      document.head.removeChild(style)
    }
  }, [isMounted, height])

  // Destroy editor on unmount
  useEffect(() => {
    return () => {
      editor?.destroy()
    }
  }, [])

  if (!editor) {
    return (
      <div 
        className="w-full border border-gray-300 rounded-lg bg-gray-50"
        style={{ minHeight: height }}
      >
        <div className="p-4 text-gray-400 text-sm">Loading editor...</div>
      </div>
    )
  }

  return (
    <div className="tiptap-wrapper">
      {/* Toolbar */}
      <div className="tiptap-toolbar">
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleBold().run()}
          className={editor?.isActive('bold') ? 'is-active' : ''}
          title="Bold"
        >
          <strong>B</strong>
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleItalic().run()}
          className={editor?.isActive('italic') ? 'is-active' : ''}
          title="Italic"
        >
          <em>I</em>
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleStrike().run()}
          className={editor?.isActive('strike') ? 'is-active' : ''}
          title="Strikethrough"
        >
          <s>S</s>
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleUnderline().run()}
          className={editor?.isActive('underline') ? 'is-active' : ''}
          title="Underline"
        >
          <u>U</u>
        </button>
        
        <div className="separator" />
        
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleHeading({ level: 1 }).run()}
          className={editor?.isActive('heading', { level: 1 }) ? 'is-active' : ''}
          title="Heading 1"
        >
          H1
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}
          className={editor?.isActive('heading', { level: 2 }) ? 'is-active' : ''}
          title="Heading 2"
        >
          H2
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}
          className={editor?.isActive('heading', { level: 3 }) ? 'is-active' : ''}
          title="Heading 3"
        >
          H3
        </button>
        
        <div className="separator" />
        
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleBulletList().run()}
          className={editor?.isActive('bulletList') ? 'is-active' : ''}
          title="Bullet List"
        >
          • List
        </button>
        <button
          type="button"
          onClick={() => editor?.chain().focus().toggleOrderedList().run()}
          className={editor?.isActive('orderedList') ? 'is-active' : ''}
          title="Numbered List"
        >
          1. List
        </button>
        
        <div className="separator" />
        
        <button
          type="button"
          onClick={() => editor?.chain().focus().unsetAllMarks().run()}
          title="Clear Formatting"
        >
          Clear
        </button>

        {onEnhance && (
          <>
            <div className="separator" />
            <button
              type="button"
              onClick={onEnhance}
              disabled={isEnhancing}
              className={isEnhancing ? 'enhancing' : ''}
              title="Enhance with AI"
              style={{ backgroundColor: isEnhancing ? '#6c757d' : '#212529', color: 'white' }}
            >
              {isEnhancing ? '✨...' : '✨ AI'}
            </button>
          </>
        )}
      </div>
      
      {/* Editor Content */}
      <EditorContent editor={editor} className="tiptap-editor" />
    </div>
  )
})

QuillEditor.displayName = 'QuillEditor'

export default QuillEditor
