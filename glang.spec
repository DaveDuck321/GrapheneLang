Name:           glang
Version:        0.1.3
Release:        %autorelease
Summary:        A Graphene front-end for LLVM

License:        AGPL-3.0-only
URL:            https://github.com/DaveDuck321/GrapheneLang
Source0:        %{url}/archive/refs/heads/dev_hatch.tar.gz
Source1:        %{url}/archive/refs/tags/bootstrap/1.tar.gz
Source2:        %{url}/archive/refs/tags/bootstrap/2.tar.gz

BuildRequires:  python3-devel
BuildRequires:  clang
BuildRequires:  lld


%global toolchain clang
%global debug_package %{nil}


%description
A Graphene front-end for LLVM (TODO).


%prep
# -n specifies the name of the top-level directory in the source archive.
%autosetup -p1 -n glang-%{version}

# Unpacks the nth source archive to the GrapheneLang-bootstrap-n subdirectory.
%setup -D -T -n glang-%{version} -a 1
%setup -D -T -n glang-%{version} -a 2

# Go back to our build directory.
cd %{_builddir}/glang-%{version}

%generate_buildrequires
%pyproject_buildrequires


%build
export BOOTSTRAP_PYTHON_BIN=%{python3}
%py3_shebang_fix glang-%{version}/GrapheneLang-bootstrap-1/glang
%py3_shebang_fix glang-%{version}/GrapheneLang-bootstrap-2/glang
%pyproject_wheel


%install
%pyproject_install
# For official Fedora packages, including files with '*' +auto is not allowed
# Replace it with a list of relevant Python modules/globs and list extra files in %%files
%pyproject_save_files '*' +auto


%check
%pyproject_check_import -t
# TODO run the full test suite.


%files -f %{pyproject_files}


%changelog
%autochangelog
