import fs from "node:fs/promises";

const tokenizerJson = JSON.parse(await fs.readFile("tokenizer.json", "utf8"));
await fs.writeFile(
  "models/whisper-small/small-tokens.txt",
  Object.entries(tokenizerJson.model.vocab)
    .concat(tokenizerJson.added_tokens.map(({ id, content }) => [content, id]))
    .toSorted(([_, a], [__, b]) => a - b)
    .map(
      ([token, id]) =>
        `${Buffer.from(token.replace(/^Ġ/, " "), "utf8").toString("base64")} ${id}`,
    )
    .join("\n"),
);
