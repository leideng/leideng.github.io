---
title: Blog
description: Blog posts by Lei Deng
kicker: Blog
header_title: Notes and Blog Posts
lead: Each post lives in the repository as a Markdown file under `_posts`.
---

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%B %-d, %Y" }}

{% if post.excerpt %}{{ post.excerpt | strip_html | strip | truncate: 180 }}{% endif %}

{% endfor %}
