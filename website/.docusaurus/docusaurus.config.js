export default {
  "title": "PPL Bench",
  "tagline": "Evaluation Framework for Probabilistic Programming Languages",
  "url": "https://pplbench.org",
  "baseUrl": "/",
  "onBrokenLinks": "throw",
  "favicon": "img/favicon.ico",
  "projectName": "pplbench",
  "organizationName": "facebookresearch",
  "themeConfig": {
    "navbar": {
      "title": "PPL Bench",
      "logo": {
        "alt": "PPL Bench logo",
        "src": "img/pplbench_logo_no_text.png"
      },
      "items": [
        {
          "to": "docs/",
          "activeBasePath": "docs",
          "label": "Docs",
          "position": "left"
        },
        {
          "to": "blog",
          "label": "Blog",
          "position": "left"
        },
        {
          "href": "https://github.com/facebookresearch/pplbench",
          "label": "GitHub",
          "position": "right"
        }
      ],
      "hideOnScroll": false
    },
    "footer": {
      "style": "dark",
      "links": [
        {
          "title": "Learn",
          "items": [
            {
              "label": "Style Guide",
              "to": "docs/"
            },
            {
              "label": "Second Doc",
              "to": "docs/doc2"
            }
          ]
        },
        {
          "title": "More",
          "items": [
            {
              "label": "Blog",
              "to": "blog"
            },
            {
              "label": "GitHub",
              "href": "https://github.com/facebookresearch/pplbench"
            }
          ]
        },
        {
          "title": "Legal",
          "items": [
            {
              "label": "Privacy",
              "href": "https://opensource.facebook.com/legal/privacy/"
            },
            {
              "label": "Terms",
              "href": "https://opensource.facebook.com/legal/terms/"
            }
          ]
        }
      ],
      "logo": {
        "alt": "Facebook Open Source Logo",
        "src": "img/oss_logo.png",
        "href": "https://opensource.facebook.com"
      },
      "copyright": "Copyright © 2020 Facebook, Inc. Built with Docusaurus."
    },
    "colorMode": {
      "defaultMode": "light",
      "disableSwitch": false,
      "respectPrefersColorScheme": false,
      "switchConfig": {
        "darkIcon": "🌜",
        "darkIconStyle": {},
        "lightIcon": "🌞",
        "lightIconStyle": {}
      }
    },
    "metadatas": []
  },
  "stylesheets": [
    "https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css"
  ],
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "path": "../docs",
          "sidebarPath": "/Users/dvinnik/work/ppl/pplbench/website/sidebars.js",
          "editUrl": "https://github.com/facebookresearch/pplbench/edit/master/website/",
          "remarkPlugins": [
            null
          ],
          "rehypePlugins": [
            null
          ]
        },
        "blog": {
          "showReadingTime": true,
          "editUrl": "https://github.com/facebookresearch/pplbench/edit/master/website/blog/"
        },
        "theme": {
          "customCss": "/Users/dvinnik/work/ppl/pplbench/website/src/css/custom.css"
        },
        "googleAnalytics": {
          "trackingID": "UA-44373548-47",
          "anonymizeIP": true
        }
      }
    ]
  ],
  "onDuplicateRoutes": "warn",
  "customFields": {},
  "plugins": [],
  "themes": [],
  "titleDelimiter": "|"
};