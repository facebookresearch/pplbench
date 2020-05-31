/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const { siteConfig, language = '' } = this.props;
    const { baseUrl, docsUrl } = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={baseUrl + 'img/pplbench_logo_top_white_text.png'} />
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('introduction.html')}>Introduction</Button>
            <Button href={'#quickstart'}>Get Started</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const { config: siteConfig, language = '' } = this.props;
    const { baseUrl } = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{ textAlign: 'center' }}>
        <h2>Get Started</h2>
        <Container>
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
              <a>Example #1, <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a>:</a>
              <MarkdownBlock>{bash`pip install pystan`}</MarkdownBlock>
              <a>Example #2, <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:</a>
              <MarkdownBlock>{bash`pip install pyjags`}</MarkdownBlock>
            </li>
            <li>
              <h4>Run PPLBench:</h4>
              <a>Example for <a href="https://mc-stan.org/users/interfaces/pystan">Stan</a> and <a href="http://mcmc-jags.sourceforge.net/">Jags</a>:</a>
              <MarkdownBlock>{bash`python PPLBench.py -m robust_regression -l jags,stan -k 5 -n 2000 -s 500 --trials 2`}</MarkdownBlock>
            </li>
            <li>
              <h4>Extra: See PPLBench's supported models, PPL implementantations and commands:</h4>
              <MarkdownBlock>{bash`python PPLBench.py -h`}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
      <div className="productShowcaseSection" style={{ textAlign: 'center' }}>
        <h2>Key Features</h2>
        <Block layout="threeColumn">
          {[
            {
              content:
                'Plug in new models and new Probabilistic Programming Language implementations of the models.',
              image: `${baseUrl}img/puzzle_pieces.svg`,
              imageAlign: 'top',
              title: 'Modular',
            },
            {
              content:
                'Using the Predictive Log Likelihood metrics makes PPLBench applicable for all types of PPLs.  ',
              image: `${baseUrl}img/compare.svg`,
              imageAlign: 'top',
              title: 'Using Predictive Log Likelihood',
            },
            {
              content:
                'Write a benchmark workflow once, reuse it across all PPLs.',
              image: `${baseUrl}img/reuse.svg`,
              imageAlign: 'top',
              title: 'Reusable',
            },
          ]}
        </Block>
      </div>
    );
    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
