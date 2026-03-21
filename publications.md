---
title: Publications
description: Recent publications for Lei Deng
kicker: Publications
header_title: Recent Publications
lead: This page highlights recent entries from the Google Scholar profile for Lei Deng. The list was refreshed against Scholar on March 21, 2026.
---

For the complete citation record, see [Google Scholar](https://scholar.google.com/citations?user=Bx2v5H8AAAAJ&hl=en). Earlier PDFs from the legacy website are still available in the repository under `/paper`.

{% assign grouped_publications = site.data.publications | group_by: "year" | sort: "name" | reverse %}
{% for year_group in grouped_publications %}
## {{ year_group.name }}

{% for publication in year_group.items %}
- **{{ publication.title }}**
  {{ publication.authors }}
  {{ publication.venue }}
{% endfor %}

{% endfor %}
