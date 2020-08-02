
#pragma once

#include "imgui/imgui.h"
#include "diligentGraphics/Math.hpp"
#include <SDL.h>
#include "OGLRender.hpp"
#include "BoidsGenerator.hpp"

class BoidsApp {
        public:
            BoidsApp();
            ~BoidsApp() = default;
            void run();
            bool isInit() const { return m_init; }

        private:
            bool initWindow();
            bool closeWindow();
            bool checkSDLStatus();
            void checkMouseState();
            void displayMainWidget();

            std::unique_ptr<Render::OGLRender> m_OGLRender;
            std::unique_ptr<Core::BoidsGenerator> m_boidsGenerator;

            SDL_Window* m_window;
            SDL_GLContext m_OGLContext;

            Math::int2 m_windowSize;
            Math::int2 m_mousePrevPos;
            ImVec4 m_backGroundColor;
            bool m_buttonLeftActivated;
            bool m_buttonRightActivated;
            bool m_init;

            int m_boxSize;
};