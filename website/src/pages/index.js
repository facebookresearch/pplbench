/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';
import exampleJson from '../../../examples/example.json';

const QuickStart = () => (
  <div id="quickstart" className={styles.gettingStartedSection}>
    <div className="container padding-vert--xl text--left">
      <div className="row">
        <div className="col">
          <h1 className="text--center">Get started in seconds</h1>
          <div>
            <ol>
              <li className="margin-top--sm">
                <h3>1. Install PPL Bench with supported PPLs:</h3>
                <CodeBlock>pip install 'pplbench[ppls]'</CodeBlock>
              </li>
              <li>
                <h3>2. Write a simple JSON config</h3>
                <p>
                  Let's store the following config into a file called{' '}
                  <a href="https://github.com/facebookresearch/pplbench/blob/master/examples/example.json">
                    <code>example.json</code>
                  </a>
                  .
                </p>
                <CodeBlock className="json">
                  {JSON.stringify(exampleJson, null, 2)}
                </CodeBlock>
              </li>
              <li>
                <h3>3. Run PPL Bench:</h3>
                <CodeBlock>pplbench example.json</CodeBlock>
                <p>
                  You should see the plots like followings in the output
                  directory:
                </p>
                <div className="container">
                  <div className="row">
                    <div className="col col--6">
                      <p>
                        <code>pll.png</code>
                      </p>
                      <img
                        alt="Full predictive log likelihood"
                        src={useBaseUrl('img/example_pystan_pll.svg')}
                      />
                    </div>
                    <div className="col col--6">
                      <p>
                        <code>pll_half.png</code>
                      </p>
                      <img
                        alt="Half predictive log likelihood"
                        src={useBaseUrl('img/example_pystan_pll_half.svg')}
                      />
                    </div>
                  </div>
                </div>
              </li>
              <li>
                <h3>4. Try PPL Bench with other configs</h3>
                <p>
                  The{' '}
                  <a href="https://github.com/facebookresearch/pplbench/tree/master/examples">
                    <code>examples</code> directory in our GitHub repo
                  </a>{' '}
                  provides a list of config files to show how PPL Bench could be
                  used. Let's try{' '}
                  <a href="https://github.com/facebookresearch/pplbench/blob/master/examples/logistic_regression.json">
                    <code>examples/logistic_regression.json</code>
                  </a>{' '}
                  to see the performance across different PPLs:
                </p>
                <p>
                  (You'll need to{' '}
                  <a href={useBaseUrl('docs/working_with_ppls#jags')}>
                    install Jags
                  </a>{' '}
                  first before running this config)
                </p>
                <CodeBlock>
                  pplbench examples/logistic_regression.json
                </CodeBlock>
                <p>The plots shoule look like the followings:</p>
                <div className="container">
                  <div className="row">
                    <div className="col col--6">
                      <p>
                        <code>pll.png</code>
                      </p>
                      <img
                        alt="Full predictive log likelihood"
                        src={useBaseUrl('img/example_blr_pll.svg')}
                      />
                    </div>
                    <div className="col col--6">
                      <p>
                        <code>pll_half.png</code>
                      </p>
                      <img
                        alt="Half predictive log likelihood"
                        src={useBaseUrl('img/example_blr_pll_half.svg')}
                      />
                    </div>
                  </div>
                </div>
              </li>
              <li>
                <h3>To see the schema of the config file used by PPL Bench</h3>
                <CodeBlock>pplbench -h</CodeBlock>
              </li>
              <li>
                <h3>To learn more about PPL Bench</h3>
                <p>
                  Read our{' '}
                  <a href={useBaseUrl('docs/introduction')}>Introduction</a>
                </p>
              </li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const features = [
  {
    title: 'Modular',
    imageUrl: 'img/puzzle_pieces.svg',
    description: (
      <>
        Plug in new models and new Probabilistic Programming Language
        implementations of the models.
      </>
    ),
  },
  {
    title: 'Using Predictive Log Likelihood',
    imageUrl: 'img/compare.svg',
    description: (
      <>
        Using the Predictive Log Likelihood metrics makes PPLBench applicable
        for all types of PPLs.
      </>
    ),
  },
  {
    title: 'Reusable',
    imageUrl: 'img/reuse.svg',
    description: (
      <>Write a benchmark workflow once, reuse it across all PPLs.</>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3 className="text--center">{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Evaluation Framework for Probabilistic Programming Languages.">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <img
            className={styles.heroLogo}
            src="img/pplbench_logo_no_text.png"
            alt="PPBench Logo"
            width="170"
          />
          <img className="imgUrl">{siteConfig.imgUrl}</img>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={'#quickstart'}>
              Get Started
            </Link>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/introduction')}>
              Introduction
            </Link>
          </div>
        </div>
      </header>
      <main>
        <div className="container padding-vert--xl text--left">
          <div className="row">
            <div className="col">
              <h1 className="text--center">Key Features</h1>
              {features && features.length > 0 && (
                <section className={styles.features}>
                  <div className="container">
                    <div className="row">
                      {features.map(({title, imageUrl, description}) => (
                        <Feature
                          key={title}
                          title={title}
                          imageUrl={imageUrl}
                          description={description}
                        />
                      ))}
                    </div>
                  </div>
                </section>
              )}
            </div>
          </div>
        </div>
      </main>
      <QuickStart />
    </Layout>
  );
}

export default Home;
