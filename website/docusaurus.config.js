/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const remarkMath = require('remark-math');
const rehypeKatex = require('rehype-katex');

module.exports = {
  title: 'PPL Bench',
  tagline: 'Evaluation Framework for Probabilistic Programming Languages',
  url: 'https://pplbench.org',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  projectName: 'pplbench',
  organizationName: 'facebookresearch',
  themeConfig: {
    announcementBar: {
      id: 'support_ukraine',
      content:
        'Support Ukraine 🇺🇦 <a target="_blank" rel="noopener noreferrer" href="https://opensource.fb.com/support-ukraine"> Help Provide Humanitarian Aid to Ukraine</a>.',
      backgroundColor: '#20232a',
      textColor: '#fff',
      isCloseable: false,
    },
    navbar: {
      title: 'PPL Bench',
      logo: {
        alt: 'PPL Bench logo',
        src: 'img/pplbench_logo_no_text.png',
      },
      items: [
        {
          to: 'docs/introduction',
          activeBasePath: 'docs/introduction',
          label: 'Docs',
          position: 'left',
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        // Please keep GitHub link to the right for consistency.
        {
          href: 'https://github.com/facebookresearch/pplbench',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: 'docs/introduction',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting_started',
            },
            // {
            //   label: "API Reference",
            //   to: "api/TO_ADD"
            // },
            // {
            //   label: 'Paper',
            //   to: 'https://arxiv.org/abs/TO_ADD',
            // },
          ],
        },

        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/facebookresearch/pplbench',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
            },
          ],
        },
      ],
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'img/oss_logo.png',
        href: 'https://opensource.facebook.com',
      },
      // Please do not remove the credits, help to publicize Docusaurus :)
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc. Built with Docusaurus.`,
    },
  },
  stylesheets: ['https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css'],
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: '../docs/',
          // homePageId: '/docs/toc',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebookresearch/pplbench/edit/main/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebookresearch/pplbench/edit/main/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        googleAnalytics: {
          trackingID: 'UA-44373548-47',
          anonymizeIP: true,
        },
      },
    ],
  ],
};
