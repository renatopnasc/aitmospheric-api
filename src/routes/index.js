const { Router } = require("express");
const routes = Router();

const soundsRoutes = require("./sounds.routes");

routes.use("/sounds", soundsRoutes);

module.exports = routes;
