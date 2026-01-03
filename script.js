const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const statusEl = document.getElementById("status");
const output = document.getElementById("output");
const debugPanel = document.getElementById("debug-panel");
const debugOutput = document.getElementById("debug-output");

const API_URL = "/query";

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#a33a3a" : "#145865";
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();

  if (!query) {
    setStatus("Please enter a query.", true);
    return;
  }

  setStatus("Analyzing...");
  output.textContent = "";
  debugOutput.textContent = "No debug info yet.";
  debugPanel.removeAttribute("open");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();

    if (!response.ok) {
      const detail = data?.detail ?? "Unknown server error.";
      throw new Error(`Server error ${response.status}: ${detail}`);
    }

    output.textContent = data.result ?? "No result returned.";
    setStatus("Done.");
  } catch (error) {
    console.error(error);
    output.textContent = "";
    setStatus("Sorry, we couldn't reach the API. Please try again.", true);
    debugOutput.textContent =
      error?.message ??
      "No debug details available. Check the console or server logs.";
    debugPanel.setAttribute("open", "open");
  }
});
