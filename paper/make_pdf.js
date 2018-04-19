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
  await engine.pandocExport({runAllCodeChunks: false})

  return process.exit();
}

main();
