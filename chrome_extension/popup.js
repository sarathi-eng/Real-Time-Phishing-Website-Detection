document.addEventListener('DOMContentLoaded', async () => {
  const urlText = document.getElementById('url-text');
  const urlContainer = document.getElementById('url-container');
  const spinner = document.getElementById('spinner');
  const resultContent = document.getElementById('result-content');
  const errorContent = document.getElementById('error-content');
  
  const verdictEl = document.getElementById('verdict');
  const reasonEl = document.getElementById('reason');
  const confidenceEl = document.getElementById('confidence');
  const errorMsgEl = document.getElementById('error-message');

  try {
    // 1. Get current active tab URL
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab || !tab.url) {
      throw new Error("Invalid URL");
    }

    const currentUrl = tab.url;

    // Reject non-http pages (like chrome://)
    if (!currentUrl.startsWith('http://') && !currentUrl.startsWith('https://')) {
      urlText.textContent = "System Page";
      throw new Error("Invalid URL");
    }

    // Shorten URL for display
    urlContainer.title = currentUrl;
    const urlObj = new URL(currentUrl);
    const shortHost = urlObj.hostname;
    urlText.textContent = shortHost.length > 30 ? shortHost.substring(0, 27) + '...' : shortHost;

    // 2. Call local API with fetch
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout

    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: currentUrl }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error("Unable to analyze (API offline)");
    }

    const data = await response.json();
    
    // Process response
    // Expecting: { verdict: "SAFE", confidence: 0.99, reason: "No threats..." }
    showResult(data.verdict, data.confidence, data.reason);

  } catch (error) {
    let msg = "Unable to analyze (API offline)";
    if (error.name === 'AbortError') {
      msg = "API timeout (Server too slow)";
    } else if (error.message.includes('Invalid URL')) {
      msg = "Invalid URL";
    } else if (error.message) {
      msg = error.message;
    }
    showError(msg);
  }

  function showResult(verdict, confidence, reason) {
    spinner.classList.add('hidden');
    resultContent.classList.remove('hidden');

    const v = (verdict || 'UNKNOWN').toUpperCase();
    
    const icons = {
      'SAFE': '✅',
      'SUSPICIOUS': '⚠️',
      'PHISHING': '🚨'
    };
    
    verdictEl.textContent = `${icons[v] || '🔍'} ${v}`;
    reasonEl.textContent = reason || 'Analyzed successfully.';
    
    if (confidence !== undefined && confidence !== null) {
      // API might return confidence directly as percentile or float
      let confPercent = parseFloat(confidence);
      if (confPercent <= 1.0) {
        confPercent = (confPercent * 100).toFixed(1);
      } else {
        confPercent = confPercent.toFixed(1);
      }
      confidenceEl.textContent = `Confidence: ${confPercent}%`;
    } else {
      confidenceEl.style.display = 'none';
    }

    // Apply color theme with specific CSS classes
    if (v === 'SAFE') {
      document.body.className = 'theme-safe';
    } else if (v === 'SUSPICIOUS') {
      document.body.className = 'theme-suspicious';
    } else if (v === 'PHISHING') {
      document.body.className = 'theme-phishing';
    }
  }

  function showError(message) {
    spinner.classList.add('hidden');
    errorContent.classList.remove('hidden');
    errorMsgEl.textContent = message;
  }
});
