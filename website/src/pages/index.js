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
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const QuickStart = () => (
  <div
    className="productShowcaseSection"
    id="quickstart"
    style={{textAlign: 'center'}}>
    <h2>Get Started</h2>
    <div>
      <ol>
        <li>
          <h4>Download PPLBench:</h4>
          <a>via git:</a>
          <MarkdownBlock>{bash`git clone https://github.com/facebookresearch/pplbench.git`}</MarkdownBlock>
        </li>
        <li>
          <h4>Install PPLBench Core:</h4>
          <MarkdownBlock>{bash`pip install -r requirements.txt`}</MarkdownBlock>
        </li>
        <li>
          <h4>Install PPLs to benchmark:</h4>
          <a>
            Example #1,{' '}
            <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a>:
          </a>
          <MarkdownBlock>{bash`pip install pystan`}</MarkdownBlock>
          <a>
            Example #2, <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:
          </a>
          <MarkdownBlock>{bash`pip install pyjags`}</MarkdownBlock>
        </li>
        <li>
          <h4>Run PPLBench:</h4>
          <a>
            Example for{' '}
            <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a> and{' '}
            <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:
          </a>
          <MarkdownBlock>{bash`python PPLBench.py -m robust_regression -l jags,stan -k 5 -n 2000 -s 500 --trials 2`}</MarkdownBlock>
        </li>
        <li>
          <h4>
            Extra: See PPLBench's supported models, PPL implementantations and
            commands:
          </h4>
          <MarkdownBlock>{bash`python PPLBench.py -h`}</MarkdownBlock>
        </li>
      </ol>
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
      <h3>{title}</h3>
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
          {/* <div className={styles.buttons}>
            <Link
              className={clsx(
                "button button--outline button--secondary button--lg",
                styles.getStarted
              )}
              to={useBaseUrl("docs/")}
            >
              Introduction
            </Link>
            <Link
              className={clsx(
                "button button--outline button--secondary button--lg",
                styles.getStarted
              )}
              to={useBaseUrl("docs/")}
            >
              Get Started
            </Link>
          </div> */}
        </div>
      </header>
      <main>
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
        {/* <QuickStart /> */}
      </main>
    </Layout>
  );
}

export default Home;
