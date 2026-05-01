# Token Probability Explorer

**Live demo:** https://eoftedal.github.io/logprobs/

## What is this?

Token Probability Explorer is a browser-based tool for visualizing the probability distribution over next tokens as a language model generates text. It runs entirely in your browser using WebGPU — no server, no API key, no data leaving your machine.

At each step the model computes a probability for every token in its vocabulary. This tool shows you the top candidates and lets you pick which one to append to the text, then recomputes the distribution from the new context. This makes it easy to explore how the model "thinks" about what comes next: which completions it considers likely, how confident it is, and how that changes as the context grows.

You can use it in **raw text** mode (feed the model a plain string) or **chat template** mode (provide a system prompt and user message, which are formatted using the model's built-in chat template before inference).

The tool supports adjustable temperature (higher = more uniform distribution, lower = more peaked) and a configurable number of top tokens to display.

## How to use it

### Requirements

- Chrome 113+ or Edge 113+ with WebGPU enabled
- If WebGPU is not available, try enabling it at `chrome://flags/#enable-unsafe-webgpu`

### Steps

1. Open https://eoftedal.github.io/logprobs/ in a supported browser.
2. **Select a model.** Models are downloaded from HuggingFace and cached in your browser on first use. The 1.5B model is a good starting point (~1 GB download).
3. Click **Load Model** and wait for the download and initialization to complete.
4. Adjust **Tokens to show** and **Temperature** to taste.
5. Choose a tab:
   - **Raw text** — type or paste any text as the starting context.
   - **Chat template** — enter a system prompt and a user message; the model's chat template is applied automatically.
6. Click **Get Next Tokens**. The top candidate tokens appear as clickable buttons, each showing its probability and a color-coded fill indicating relative likelihood.
7. Click a token to append it to the text and recompute the distribution.
8. Use **Undo** to step back one token, or **Reset** to start over with new input.

To use a different model, reload the page.
