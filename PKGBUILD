pkgname=fex-wtf
pkgver=2608
pkgrel=1
pkgdesc='Tool for seeing FEX emulator stats'
arch=(aarch64)
url='https://github.com/Sonicadvance1/what-the-FEX'
license=('MIT')
depends=(ncurses)
makedepends=(cmake ninja)
source=('git+https://github.com/Sonicadvance1/what-the-FEX.git')
sha256sums=('SKIP')

build() {
  cd what-the-FEX
  rm -rf build
  mkdir build
  cd build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
  ninja
}

package() {
  cd what-the-FEX/build
  DESTDIR="$pkgdir" ninja install
}
