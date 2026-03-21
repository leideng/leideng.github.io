---
title: Publications
description: Recent publications for Lei Deng
---

<section class="page-header">
  <p class="section-kicker">Publications</p>
  <h1>Recent Publications</h1>
  <p class="lead">
    This page highlights recent entries from the Google Scholar profile for Lei Deng.
    The list was refreshed against Scholar on March 21, 2026.
  </p>
  <p>
    For the complete citation record, see
    <a href="https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en">Google Scholar</a>.
    Earlier PDFs from the legacy website are still available in the repository under <code>/paper</code>.
  </p>
</section>

<section class="markdown-body">
  {% assign grouped_publications = site.data.publications | group_by: "year" | sort: "name" | reverse %}
  {% for year_group in grouped_publications %}
  <h2 class="pub-year">{{ year_group.name }}</h2>
  {% for publication in year_group.items %}
  <article class="pub-item">
    <h3>{{ publication.title }}</h3>
    <p>{{ publication.authors }}</p>
    <p class="pub-meta">{{ publication.venue }}</p>
  </article>
  {% endfor %}
  {% endfor %}
</section>
