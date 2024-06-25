const tf = require("@tensorflow/tfjs-node");

const InputError = require("../exceptions/InputError");

async function predictClass(model, image) {
  try {
    const tensor = tf.node
      .decodeImage(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    console.log("score: ", score);
    console.log("confidenceScore: ", confidenceScore);

    const label = confidenceScore > 50 ? "Cancer" : "Non-cancer";

    let suggestion;

    if (label === "Cancer") {
      suggestion =
        "Mohon melakukan pemeriksaan lebih lanjut ke dokter Spesialis, jangan panik! selalu ada jalan untuk setiap masalah..";
    } else {
      suggestion =
        "Alhamdulillah, jangan lupa tetap jaga kesehatan!.";
    }
    return {
      confidenceScore,
      label,
      suggestion: suggestion,
    };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClass;