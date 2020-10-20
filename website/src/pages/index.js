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

const QuickStart = () => (
  <div id="quickstart" className={styles.gettingStartedSection}>
    <div className="container padding-vert--xl text--left">
      <div className="row">
        <div className="col">
          <h1 className="text--center">Get started in seconds</h1>

          <div>
            <ol>
              <li className="margin-top--sm">
                <h3>1. Download PPL Bench:</h3>
                <CodeBlock>
                  git clone https://github.com/facebookresearch/pplbench.git
                </CodeBlock>
              </li>
              <li>
                <h3>2. Install PPL Bench:</h3>
                <CodeBlock>{`pip install -r requirements.txt`}</CodeBlock>
              </li>
              <li>
                <h3>3. Install PPLs to benchmark:</h3>
                <a>
                  Example #1,{' '}
                  <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a>
                  :
                </a>
                <CodeBlock>{`pip install pystan`}</CodeBlock>
                <a>
                  Example #2,{' '}
                  <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:
                </a>
                <CodeBlock>{`pip install pyjags`}</CodeBlock>
              </li>
              <li>
                <h3>4. Run PPL Bench:</h3>
                <a>
                  Example for{' '}
                  <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a>{' '}
                  and <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:
                </a>
                <CodeBlock>{`python PPLBench.py -m robust_regression -l jags,stan -k 5 -n 2000 -s 500 --trials 2`}</CodeBlock>
              </li>
              <li>
                <h3>
                  <i>Extra:</i> See PPL Bench's supported models, PPL
                  implementations and commands:
                </h3>
                <CodeBlock>{`python PPLBench.py -h`}</CodeBlock>
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
