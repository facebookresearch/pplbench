The PPLBench website was created with [Docusaurus](https://docusaurus.io/).
FontAwesome icons were used under the
[Creative Commons Attribution 4.0 International](https://fontawesome.com/license).

## Building

You need [Node](https://nodejs.org/en/) >= 8.x and
[Yarn](https://yarnpkg.com/en/) >= 1.5 in order to build the pplbench website.

Switch to the `website` dir from the project root and start the server:
```bash
cd website
yarn
yarn start
```

Open http://localhost:3000 (if doesn't automatically open).

Anytime you change the contents of the page, the page should auto-update.
<!-- 
#### Generating a static build

To generate a static build of the website in the `website/build` directory, run
```bash
./scripts/build_docs.sh -b
``` -->

## Publishing

The site is hosted on GitHub pages, via the `gh-pages` branch of the PPLBench
[GitHub repo](https://github.com/facebookresearch/pplbench/tree/gh-pages).
The website is automatically built and published from CircleCI - see the
[config file](https://github.com/facebookresearch/pplbench/blob/master/.circleci/config.yml)
for details.
