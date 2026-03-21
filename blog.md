---
title: Blog
description: Blog posts by Lei Deng
---

<section class="page-header">
  <p class="section-kicker">Blog</p>
  <h1>Notes and Blog Posts</h1>
  <p class="lead">Each post lives in the repository as a Markdown file under <code>_posts</code>.</p>
</section>

<section class="markdown-body">
  {% for post in site.posts %}
  <article class="post-card">
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p class="post-date">{{ post.date | date: "%B %-d, %Y" }}</p>
    {% if post.excerpt %}
    <p>{{ post.excerpt | strip_html | truncate: 180 }}</p>
    {% endif %}
  </article>
  {% endfor %}
</section>
