import React from 'react';

const ResultPage = ({ result }) => {
  return (
    <div className="ResultPage">
      <h2>Prediction Results</h2>
      <div>
        <h3>Prediction:</h3>
        <p>Alzheimer's Risk: {result.prediction["Alzheimer's_risk"]}</p>
        <p>Parkinson's Risk: {result.prediction["Parkinson's_risk"]}</p>
        <p>Other Disorders: {result.prediction["other_disorders"]}</p>
      </div>
      <div>
        <h3>Diagnostic Report:</h3>
        <p>MRI Analysis: {result.diagnostic_report.MRI_analysis}</p>
        <ul>
          {result.diagnostic_report.recommendations.map((rec, index) => (
            <li key={index}>{rec}</li>
          ))}
        </ul>
      </div>
      <div>
        <h3>Visualized Results:</h3>
        <a href={result.visualized_results} target="_blank" rel="noopener noreferrer">View Detailed Analysis</a>
      </div>
    </div>
  );
};

export default ResultPage;
