require("dotenv/config");

const express = require("express");
const app = express();
const cors = require("cors");

const routes = require("./routes");

app.use(cors());

app.use(express.json());
app.use(routes);

const PORT = process.env.PORT || 3333;
app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));
