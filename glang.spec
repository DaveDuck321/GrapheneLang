Name:           glang
Version:        0.2.0
Release:        %autorelease
Summary:        A Graphene front-end for LLVM

License:        AGPL-3.0-only
URL:            https://github.com/DaveDuck321/GrapheneLang
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz
Source1:        %{url}/archive/refs/tags/bootstrap/1.tar.gz
Source2:        %{url}/archive/refs/tags/bootstrap/2.tar.gz

BuildRequires:  python3-devel
BuildRequires:  clang
BuildRequires:  lld


%global toolchain clang
%global debug_package %{nil}

%define source_dir GrapheneLang-%{version}

%description
A Graphene front-end for LLVM (TODO).


%prep
# -n specifies the name of the top-level directory in the source archive.
%autosetup -p1 -n %{source_dir}

# Unpacks the nth source archive to the GrapheneLang-bootstrap-n subdirectory.
%setup -D -T -n %{source_dir} -a 1
%setup -D -T -n %{source_dir} -a 2

# Go back to our build directory.
cd %{_builddir}/%{source_dir}

# Initializing Lark with strict=True requires the interegular package. Turn off
# strict mode so that we can run the tests without it.
sed -i "s/strict=True/strict=False/" tests/test_config_parser.py


%generate_buildrequires
%pyproject_buildrequires


%build
%pyproject_wheel


%install
%pyproject_install
# For official Fedora packages, including files with '*' +auto is not allowed
# Replace it with a list of relevant Python modules/globs and list extra files in %%files
%pyproject_save_files '*' +auto


%check
%pyproject_check_import -t
%{py3_test_envvars} %{python3} tests/run_tests.py


%files -f %{pyproject_files}


%changelog
%autochangelog
