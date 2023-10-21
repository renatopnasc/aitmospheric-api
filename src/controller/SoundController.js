// const { PythonShell } = require("python-shell");

// class SoundController {
//   create(request, response) {
//     const options = {
//       pythonPath: "venv/bin/python",
//     };
//     PythonShell.run("src/services/run_model.py", options)
//       .then((message) => {
//         // Quando o processo terminar, realizar o download do arquivo .wav na pasta assets/audios
//         response.download("aitmosphere.wav");

//         return response.json({});
//       })
//       .catch((error) => {
//         console.log(error);
//         return;
//       });
//   }
// }

// module.exports = SoundController;

const { PythonShell } = require("python-shell");
const path = require("path");

class SoundController {
  create(request, response) {
    const options = {
      pythonPath: "venv/bin/python",
    };

    PythonShell.run("src/services/run_model.py", options)
      .then((message) => {
        // Quando o processo terminar, realizar o download do arquivo .wav na pasta assets/audios

        response.download(
          "assets/audios/aitmosphere.wav",
          "aitmosphere.wav",
          (err) => {
            if (err) {
              console.log("Erro ao realizar o download do arquivo:", err);
              return response
                .status(500)
                .json({ error: "Erro ao realizar o download do arquivo" });
            }
          }
        );
      })
      .catch((error) => {
        console.log(error);
        return response
          .status(500)
          .json({ error: "Erro ao executar o processo Python" });
      });
  }
}

module.exports = SoundController;
