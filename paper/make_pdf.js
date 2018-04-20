// node.js
const mume = require('@shd101wyy/mume')

async function main() {
  await mume.init()

  const engine = new mume.MarkdownEngine({
    filePath: "paper.md",
    config: {
    }
  })

 	// pandoc export
  try {
    await engine.pandocExport({runAllCodeChunks: false});
  } catch (error)  {
    console.log("error", error);
  }

  return process.exit();

}

main();
