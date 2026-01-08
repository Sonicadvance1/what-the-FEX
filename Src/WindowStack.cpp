#include "WindowStack.hpp"

#include <ncurses.h>
#include <unistd.h>

namespace WTF {
static void exit_screen(const char* format = nullptr, ...) {
  refresh();
  endwin();

  if (format != nullptr) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
  }
  _exit(0);
}

void WinStack::RequestNewHeight(int StackID, int Height) {
  for (auto &Win : Stack) {
    if (Win.StackID != StackID) continue;
    Win.Props.Height = Height;
    break;
  }
  NewHeightRequested = true;
}

void WinStack::UpdateWindowDimensions() {
  auto width = COLS;
  auto height = LINES;

  if (!NewHeightRequested && width == WindowWidth && height == WindowHeight) {
    return;
  }

  WindowWidth = width;
  WindowHeight = height;

  size_t y = 0;
  for (auto &Win : Stack) {
    bool NeedsUpdatedCoords {};
    auto win_x = getparx(Win.win);
    auto win_y = getpary(Win.win);

    const auto win_height = getmaxy(Win.win);
    auto win_width = getmaxx(Win.win);

    if (win_y != y || win_height != Win.Props.Height) {
      NeedsUpdatedCoords = true;
      win_y = y;
    }

    y += Win.Props.Height;

    if ((win_x + win_width) != WindowWidth) {
      NeedsUpdatedCoords = true;
      win_width = WindowWidth - win_x;
    }

    if (NeedsUpdatedCoords) {
      if (Resize(Win.win, win_width, win_height) != OK) exit_screen("Couldn't resize: %d\n", Win.StackID);
      if (Move(Win.win, win_x, win_y) != OK) exit_screen("Couldn't move: %d -> %d %d\n", Win.StackID, win_x, win_y);
    }
  }

  NewHeightRequested = false;
}

int WinStack::AddToStack(WindowCallback callback, WINDOW* win, void* user_data, const Properties &props) {
  int ID = StackID;
  StackID++;
  Stack.emplace_back(WindowData {
    callback, win, user_data, ID,
    props
  });

  return ID;
}

void WinStack::RunStack() {
  for (const auto& Member : Stack) {
    Member.callback(Member.win, Member.user_data);
  }
}

int WinStack::Resize(WINDOW* win, int width, int height) {
  return wresize(win, height, width);
}

int WinStack::Move(WINDOW* win, int x, int y) {
  return mvderwin(win, y, x);
}

}
