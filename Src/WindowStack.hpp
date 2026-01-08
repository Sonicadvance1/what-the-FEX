#pragma once
#include <vector>

typedef struct _win_st WINDOW;

namespace WTF {
class WinStack final {
  public:
    struct Properties {
      int Height {-1};
    };
    using WindowCallback = void (*)(WINDOW *win, void* user_data);
    void UpdateWindowDimensions();
    int AddToStack(WindowCallback callback, WINDOW* win, void* user_data, const Properties &props);
    void RunStack();

    void RequestNewHeight(int StackID, int Height);

  private:
    int StackID {};
    struct WindowData {
      WindowCallback callback;
      WINDOW* win;
      void* user_data;
      int StackID;
      Properties Props;
    };
    int WindowWidth {};
    int WindowHeight {};

    std::vector<WindowData> Stack;

    int Resize(WINDOW* win, int width, int height);
    int Move(WINDOW* win, int x, int y);
    bool NewHeightRequested = false;
};
}
