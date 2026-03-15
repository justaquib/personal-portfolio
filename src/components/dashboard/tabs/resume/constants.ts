// Resume Builder Constants

// Template definitions
export const TEMPLATES = [
  {
    id: 'modern',
    name: 'Modern',
    description: 'Clean and professional with a contemporary look',
    preview: 'bg-gradient-to-r from-gray-600 to-gray-800'
  },
  {
    id: 'classic',
    name: 'Classic',
    description: 'Traditional resume format, perfect for corporate jobs',
    preview: 'bg-gradient-to-r from-gray-600 to-gray-800'
  },
  {
    id: 'minimal',
    name: 'Minimal',
    description: 'Simple and elegant with ample white space',
    preview: 'bg-gradient-to-r from-blue-500 to-cyan-500'
  },
  {
    id: 'creative',
    name: 'Creative',
    description: 'Stand out with a unique and memorable design',
    preview: 'bg-gradient-to-r from-orange-500 to-red-500'
  }
]

// Skill suggestions for resume
export const SKILL_SUGGESTIONS = [
  // Programming Languages
  'JavaScript', 'TypeScript', 'Python', 'Java', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'PHP', 'Swift', 'Kotlin',
  // Frontend
  'React', 'Vue.js', 'Angular', 'Next.js', 'Nuxt.js', 'Svelte', 'HTML', 'CSS', 'Tailwind CSS', 'Sass',
  // Backend
  'Node.js', 'Express.js', 'Django', 'Flask', 'Spring Boot', 'Ruby on Rails', 'Laravel', 'ASP.NET',
  // Databases
  'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'SQLite',
  // Cloud & DevOps
  'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform', 'Jenkins', 'CI/CD',
  // Tools & Others
  'Git', 'GitHub', 'GitLab', 'Linux', 'REST API', 'GraphQL', 'Agile', 'Scrum', 'TDD'
]

// Section definitions for drag and drop
export const SECTION_ORDER = [
  { id: 'personal', label: 'Personal Information', icon: 'User' },
  { id: 'summary', label: 'Professional Summary', icon: 'FileText' },
  { id: 'experience', label: 'Work Experience', icon: 'Briefcase' },
  { id: 'education', label: 'Education', icon: 'GraduationCap' },
  { id: 'skills', label: 'Skills', icon: 'Code' },
  { id: 'projects', label: 'Projects', icon: 'Folder' },
  { id: 'certifications', label: 'Certifications', icon: 'Award' },
  { id: 'websites', label: 'Websites & Portfolio', icon: 'Globe' },
  { id: 'languages', label: 'Languages', icon: 'Globe' },
]
