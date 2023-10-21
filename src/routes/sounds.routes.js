const { Router } = require("express");

const soundsRouter = Router();

const SoundsController = require("../controller/SoundController");
const soundController = new SoundsController();

soundsRouter.get("/", soundController.create);

module.exports = soundsRouter;
