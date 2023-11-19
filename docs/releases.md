# Releases

## Versioning

Make sure you update the version in both [glang.spec](/glang.spec) and [pyproject.toml](/pyproject.toml)

## PyPI

- Run `rm dist/*` to delete stale builds
- Run `hatch build` to build your working tree
- Run `hatch publish` to push _all_ files in the `dist/` directory

## RPM

### Building locally

It's always a good idea to build and test the RPM before releasing.

- Run `rpmdev-setuptree` to create the expected file tree in `~/rpmbuild`
- Copy [the spec file](/glang.spec) to `~/rpmbuild/SPECS/glang.spec`
- Run `spectool -g -R SPECS/glang.spec` to download the specified sources (you could also provide these manually, but it's best to check that they're correct)
- Run `rpmbuild -bb SPECS/glang.spec` to build and test the RPM

### Releasing

If the RPM builds succesfully, then
- Bump the version in [glang.spec](/glang.spec)
- Commit locally on the `main` branch
- Run `git tag v0.x.y` to tag your commit
- Push `main` *and* the new tag
- Fedora Copr will pick up the build automatically; you can check its progress [here](https://copr.fedorainfracloud.org/coprs/antros/graphene/builds/)
