import React from 'react';

const ResultPage = ({ result }) => {
    return (
        <div>
            <h2>Prediction Results</h2>
            <div>
                <h3>Prediction:</h3>
                <p>{result.prediction}</p>
            </div>
            <div>
                <h3>Diagnostic Report:</h3>
                <p>{result.diagnostic_report}</p>
            </div>
            <div>
                <h3>Visualized Results:</h3>
                {/* Link to visualized results (if available) */}
                <a href={result.visualized_results} target="_blank" rel="noopener noreferrer">
                    View Detailed Analysis
                </a>
            </div>
        </div>
    );
};

export default ResultPage;
