import Image from "next/image";
import Link from "next/link";
import {
  ArrowDown,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Code2,
  Layers3,
  Mail,
  Sparkles,
} from "lucide-react";

const projects = [
  {
    number: "01",
    title: "Zoma.ai",
    description: "Scoping intelligence for professional services sales teams.",
    category: "ZOMA.AI ↗",
    href: "https://zoma.ai",
    tone: "project-card--blue",
    visual: "zoma",
  },
  {
    number: "02",
    title: "Aasvana",
    description: "A website built for Aasvana.",
    category: "AASVANA.COM ↗",
    href: "https://aasvana.com",
    tone: "project-card--peach",
    visual: "aasvana",
  },
  {
    number: "03",
    title: "Citrux.ai",
    description: "Enterprise AI, from compute infrastructure to governed AI systems.",
    category: "CITRUX.AI ↗",
    href: "https://citrux.ai",
    tone: "project-card--green",
    visual: "citrux",
  },
  {
    number: "04",
    title: "FreEMI · Ntactus",
    description: "A digital marketplace for financial products and services, by Ntactus.",
    category: "FREEMI.IN ↗",
    href: "http://freemi.in/",
    tone: "project-card--peach",
    visual: "freemi",
  },
  {
    number: "05",
    title: "Andaman Trip Maker",
    description: "A travel site for discovering and planning Andaman island getaways.",
    category: "ANDAMANTRIPMAKER.IN ↗",
    href: "https://andamantripmaker.in",
    tone: "project-card--lilac",
    visual: "andaman",
  },
  {
    number: "06",
    title: "Accolade MS",
    description: "A sports and tournament platform for matches, leagues, and members.",
    category: "ACCOLADEMS.COM ↗",
    href: "https://accoladems.com",
    tone: "project-card--blue",
    visual: "accolade",
  },
];

const prototypes = [
  ["Market Intelligence Dashboard", "/prototypes/market-intelligence"],
  ["Collaboration Board", "/prototypes/collab-board"],
  ["AI Document Workspace", "/prototypes/doc-reader"],
  ["Code Playground", "/prototypes/code-playground"],
] as const;

const reviewDrafts = [
  {
    initials: "R",
    name: "Rozy",
    company: "Andaman Trip Maker",
    quote:
      "Aquib understood what we needed and helped turn it into a website that makes planning an Andaman trip feel simple. The process was clear, and he was easy to work with.",
  },
  {
    initials: "I",
    name: "Ilia",
    company: "Zoma",
    quote:
      "Aquib brought care to the details and worked closely with our team to translate a complex product into a clear, polished experience.",
  },
  {
    initials: "D",
    name: "Dagan",
    company: "Citrux.ai",
    quote:
      "Aquib helped bring a complex AI platform to life with a thoughtful interface and a smoother experience across the product.",
  },
  {
    initials: "F",
    name: "Faraz",
    company: "Accolade MS",
    quote:
      "Aquib made the platform easier to navigate and gave our members a clearer way to follow matches and tournaments.",
  },
];

const experience = [
  {
    company: "Codelogicx",
    title: "Sr. Frontend Developer",
    years: "2023 — Present",
    detail:
      "Led development of data visualization tools using React and Chart.js, with a focus on performance and user experience.",
  },
  {
    company: "Collegify",
    title: "Software Engineer",
    years: "2022 — 2023",
    detail:
      "Built Google Calendar integrated apps and modular UI systems for productivity.",
  },
  {
    company: "Abdullah Dossary Group",
    title: "IT Engineer",
    years: "2019 — 2022",
    detail:
      "Delivered web applications across finance, education, and manufacturing.",
  },
  {
    company: "Earlier roles",
    title: "Full Stack Developer",
    years: "2017 — 2019",
    detail:
      "Built custom web solutions at Arthadut Pvt Ltd and Andaman Live Holidays.",
  },
];

const services = [
  {
    number: "01",
    icon: Code2,
    title: "Frontend engineering",
    description:
      "Fast, maintainable web interfaces built with modern frontend tools.",
    tags: "React · Next.js · TypeScript",
  },
  {
    number: "02",
    icon: Layers3,
    title: "Product UI & experiences",
    description:
      "Thoughtful user flows and clear interfaces for complex products.",
    tags: "UI · UX · Design systems",
  },
  {
    number: "03",
    icon: Sparkles,
    title: "Prototypes & experiments",
    description:
      "Useful concepts brought to life, from small interactions to working tools.",
    tags: "Prototyping · Motion · AI",
  },
];

const process = [
  ["01", "Understand", "Get clear on the people, problem, and outcome."],
  ["02", "Shape", "Map the experience and choose a practical direction."],
  ["03", "Build", "Create, test, and refine the product together."],
  ["04", "Improve", "Learn from use and keep making the experience better."],
];

const faqs = [
  {
    question: "What kind of work are you looking for?",
    answer:
      "I’m open to frontend and product engineering roles where I can help build clear, reliable digital experiences.",
  },
  {
    question: "What do you work with?",
    answer:
      "My work includes React, Next.js, TypeScript, data visualization, and interactive product interfaces.",
  },
  {
    question: "Can I see more of your work?",
    answer:
      "Yes. The selected work on this page links to live prototypes, and the prototypes page has more experiments.",
  },
  {
    question: "How can we get in touch?",
    answer:
      "Email me at developer@justaquib.com. I’m happy to talk about a role, a project, or an idea.",
  },
];

function SectionHeading({
  eyebrow,
  title,
  note,
}: {
  eyebrow: string;
  title: string;
  note?: string;
}) {
  return (
    <div className="section-heading">
      <p className="eyebrow">{eyebrow}</p>
      <div className="section-heading__row">
        <h2>{title}</h2>
        {note && <p className="section-heading__note">{note}</p>}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main className="portfolio-page">
      <header className="site-header">
        <Link className="wordmark" href="#top" aria-label="Just Aquib home">
          JustAquib<span>.</span>
        </Link>
        <nav className="site-nav" aria-label="Main navigation">
          <Link href="#work">Work</Link>
          <Link href="#services">What I do</Link>
          <Link href="#about">About</Link>
          <Link className="nav-contact" href="#contact">
            Let&apos;s talk <ArrowUpRight size={15} />
          </Link>
        </nav>
      </header>

      <section className="hero section-shell" id="top">
        <div className="hero__copy">
          <p className="eyebrow hero__eyebrow">
            <span className="availability-dot" /> Software engineer & designer
          </p>
          <h1>
            I build digital
            <br />
            things with <span>purpose.</span>
          </h1>
          <div className="hero__bottom">
            <p>
              I&apos;m Aquib — an engineer who cares about the details. I turn
              complex ideas into clear, useful web experiences.
            </p>
            <div className="hero__actions">
              <Link className="button button--dark" href="#work">
                Explore my work <ArrowDownRight size={16} />
              </Link>
              <a
                className="text-link"
                href="/assets/docs/Aquib_Shahbaz_Resume.pdf"
                download
              >
                Download résumé <ArrowDown size={15} />
              </a>
            </div>
          </div>
        </div>
        <div className="hero__portrait-wrap" aria-label="Floating 3D Aquib Shahbaz signature sculpture">
          <div className="hero-scene" role="img" aria-label="Dimensional 3D sculpture of Aquib Shahbaz's signature with orbiting rings">
            <div className="hero-scene__glow" />
            <div className="hero-orbit hero-orbit--one"><i /></div>
            <div className="hero-orbit hero-orbit--two"><i /></div>
            <div className="hero-orbit hero-orbit--three" />
            <div className="hero-sculpture">
              <div className="hero-sculpture__back" />
              <div className="hero-sculpture__face">
                <span className="hero-sculpture__monogram">JA</span>
                <Image
                  src="/assets/imgs/justaquib.png"
                  alt=""
                  width={1016}
                  height={445}
                  priority
                  className="hero-sculpture__signature"
                />
                <span className="hero-sculpture__caption">ENGINEER · DESIGNER · BUILDER</span>
              </div>
            </div>
            <span className="hero-scene__spark hero-scene__spark--one">✳</span>
            <span className="hero-scene__spark hero-scene__spark--two">✧</span>
            <span className="hero-scene__orbit-label">CREATIVE<br />ENGINEERING</span>
          </div>
          <span className="portrait-caption">Based in India · Working worldwide</span>
        </div>
        <div className="hero__index"><span>01</span> — INTRODUCTION</div>
      </section>

      <section className="intro-strip">
        <p>
          Good products make hard things feel simple. I bring engineering,
          curiosity, and a design-minded eye to making them.
        </p>
        <span>Scroll to explore <ArrowDown size={15} /></span>
      </section>

      <section className="section-shell work-section" id="work">
        <SectionHeading
          eyebrow="Selected client projects · 01 — 06"
          title="Selected work"
          note="A few of the websites and products I’ve helped bring to life. Select a project to visit its live site."
        />
        <div className="project-grid">
          {projects.map((project) => (
            <a
              className={`project-card ${project.tone}`}
              href={project.href}
              target="_blank"
              rel="noreferrer"
              key={project.number}
            >
              <div className={`project-visual project-visual--${project.visual}`}>
                <span className="project-visual__number">{project.number}</span>
                <div className={`site-preview site-preview--${project.visual}`} aria-hidden="true">
                  {project.visual === "zoma" && <><div className="site-preview__nav"><b>ZOMA</b><span>Product　 Customers　 About</span></div><div className="site-preview__hero"><small>SCOPING INTELLIGENCE</small><strong>Scope right,<br />scope with<br /><i>intelligence.</i></strong><span>✳</span></div><div className="site-preview__footer">BETTER SCOPES. STRONGER GROWTH.</div></>}
                  {project.visual === "aasvana" && <><div className="site-preview__nav"><b>AASVANA</b><span>Home　 About　 Contact</span></div><div className="aasvana-art"><span>DESIGN<br />THAT FEELS<br /><i>like you.</i></span><div /></div><div className="site-preview__footer">A THOUGHTFUL DIGITAL EXPERIENCE</div></>}
                  {project.visual === "citrux" && <><div className="site-preview__nav"><b>CITRUX<span> AI</span></b><span>Engines　 Products　 Contact</span></div><div className="citrux-art"><div className="citrux-art__core">CORA<br /><small>ENTERPRISE AI OS</small></div><i /><i /><i /><b>BUILD THE AI FACTORY.</b></div><div className="site-preview__footer">INTELLIGENCE · INFRASTRUCTURE · GOVERNANCE</div></>}
                  {project.visual === "freemi" && <><div className="site-preview__nav"><b>FreEMI</b><span>Loans　 Credit　 Tools</span></div><div className="freemi-art"><small>FINANCE, MADE MORE ACCESSIBLE</small><strong>Find the right<br /><i>way forward.</i></strong><div><span>PERSONAL LOAN</span><b>→</b></div><div><span>BUSINESS FINANCE</span><b>→</b></div><div><span>PLAN & CALCULATE</span><b>→</b></div></div><div className="site-preview__footer">NTACTUS · FINTECH PLATFORM</div></>}
                  {project.visual === "andaman" && <><div className="site-preview__nav"><b>ANDAMAN<br />TRIP MAKER</b><span>Packages　 Destinations</span></div><div className="andaman-art"><span>FIND YOUR<br /><i>island escape</i></span><div className="andaman-art__sun" /><div className="andaman-art__sea" /></div><div className="site-preview__footer">CURATED JOURNEYS · ISLAND STORIES</div></>}
                  {project.visual === "accolade" && <><div className="site-preview__nav"><b>ACCOLADE<span> MS</span></b><span>Matches　 Members　 Live</span></div><div className="accolade-art"><small>UPCOMING MATCH</small><div><b>AC</b><span>VS</span><b>FC</b></div><strong>LEAGUE · 18:30</strong></div><div className="site-preview__footer">PLAY TOGETHER. GO FURTHER.</div></>}
                </div>
                <span className="project-visual__arrow"><ArrowUpRight size={20} /></span>
              </div>
              <div className="project-card__info">
                <div><h3>{project.title}</h3><p>{project.description}</p></div>
                <span className="project-card__stack">{project.category}</span>
              </div>
            </a>
          ))}
        </div>
        <div className="prototype-row">
          <div><p className="eyebrow">More experiments</p><h3>Things I build for fun.</h3></div>
          <div className="prototype-row__links">
            {prototypes.map(([title, href]) => <Link href={href} key={href}>{title}<ArrowUpRight size={13} /></Link>)}
            <Link className="prototype-row__all" href="/prototypes">All prototypes <ArrowRight size={14} /></Link>
          </div>
        </div>
      </section>

      <section className="services-section" id="services">
        <div className="section-shell">
          <SectionHeading
            eyebrow="How I can contribute · 02"
            title="Good ideas, made real."
            note="I bring product thinking and hands-on engineering together, from the first sketch to the final details."
          />
          <div className="services-grid">
            {services.map(({ number, icon: Icon, title, description, tags }) => (
              <article className="service-card" key={number}>
                <div className="service-card__top"><span>{number}</span><Icon size={22} strokeWidth={1.5} /></div>
                <h3>{title}</h3>
                <p>{description}</p>
                <span className="service-card__tags">{tags}</span>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section-shell about-section" id="about">
        <div className="about-section__intro">
          <p className="eyebrow">A little about me · 03</p>
          <h2>Engineer by logic.<br /><span>Designer by heart.</span></h2>
        </div>
        <div className="about-section__body">
          <p className="about-lede">
            Hi, I&apos;m Aquib Shahbaz — a software engineer and creative frontend
            developer focused on building smooth, scalable web apps with
            human-friendly design.
          </p>
          <p>
            I&apos;ve worked across India and the Middle East, crafting tools from
            dashboards to internal platforms. I love clean, clever solutions
            that make everyday work a little easier.
          </p>
          <div className="about-links">
            <a href="mailto:developer@justaquib.com"><Mail size={16} /> Say hello</a>
            <Link href="/prototypes">Explore experiments <ArrowUpRight size={15} /></Link>
          </div>
        </div>
        <div className="experience-block">
          <div className="experience-block__heading">
            <p className="eyebrow">Where I&apos;ve been</p>
            <p>Experience across product, engineering, and technology.</p>
          </div>
          <div className="experience-list">
            {experience.map((item) => (
              <article className="experience-item" key={item.company}>
                <div><h3>{item.company}</h3><p>{item.title}</p></div>
                <p className="experience-item__detail">{item.detail}</p>
                <span>{item.years}</span>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="process-section">
        <div className="section-shell">
          <SectionHeading
            eyebrow="A thoughtful way to work · 04"
            title="From first thought to finished."
          />
          <div className="process-grid">
            {process.map(([number, title, description]) => (
              <article className="process-step" key={number}>
                <span>{number}</span><h3>{title}</h3><p>{description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section-shell proof-section" id="testimonials">
        <SectionHeading
          eyebrow="Client feedback · 05"
          title="What people say"
          // note="Draft wording for review. Publish each quote only after the person confirms it."
        />
        <div className="review-grid">
          {reviewDrafts.map((review) => (
            <article className="review-card" key={review.name}>
              <div className="review-card__top">
                <span className="review-card__initial">{review.initials}</span>
                {/* <span className="review-card__draft">DRAFT · CONFIRM QUOTE</span> */}
              </div>
              <span className="review-card__quote-mark">“</span>
              <blockquote>{review.quote}</blockquote>
              <div className="review-card__person">
                <strong>{review.name}</strong>
                <span>{review.company}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="pricing-section" id="pricing">
        <div className="section-shell pricing-layout">
          <div>
            <p className="eyebrow">Working together · 06</p>
            <h2>Clear scope.<br /><span>Thoughtful work.</span></h2>
          </div>
          <div className="pricing-card">
            <span className="pricing-card__label">PROJECT PRICING</span>
            <h3>Let&apos;s talk about the work.</h3>
            <p>
              Pricing depends on the role, project scope, and timeline. I share
              a clear estimate after we understand what you need.
            </p>
            {/* <div className="pricing-card__placeholder">Rate placeholder · to be confirmed</div> */}
            <a className="button button--dark" href="mailto:developer@justaquib.com?subject=Let%27s%20work%20together">
              Discuss an opportunity <ArrowUpRight size={16} />
            </a>
          </div>
        </div>
      </section>

      <section className="section-shell faq-section" id="faq">
        <SectionHeading
          eyebrow="A few answers · 07"
          title="Questions, answered."
          note="Something else on your mind? Just send me a note."
        />
        <div className="faq-list">
          {faqs.map((item, index) => (
            <details className="faq-item" key={item.question} open={index === 0}>
              <summary><span>{item.question}</span><span className="faq-item__plus">+</span></summary>
              <p>{item.answer}</p>
            </details>
          ))}
        </div>
      </section>

      <footer className="contact-section" id="contact">
        <div className="section-shell contact-inner">
          <p className="eyebrow"><span className="availability-dot" /> Open to the right opportunity</p>
          <h2>Have something<br />good in mind?</h2>
          <p className="contact-copy">I&apos;d love to hear what you&apos;re building.</p>
          <a className="contact-email" href="mailto:developer@justaquib.com">Let&apos;s talk <ArrowUpRight size={26} /></a>
          <div className="contact-bottom">
            <Link className="wordmark" href="#top">JustAquib<span>.</span></Link>
            <span>© {new Date().getFullYear()} Aquib Shahbaz</span>
            <a href="#top">Back to top ↑</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
